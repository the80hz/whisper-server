"""Tests for upload limits, duration-based timeouts, channel splitting and hotwords.

The audio helpers and the model are stubbed, so nothing here needs a GPU, ffmpeg
or a downloaded model.
"""

import asyncio
import io
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
import numpy as np
import pytest

from whisper_server import server
from whisper_server.audio import AudioInfo


def _word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end, probability=0.9)


def _segment(start: float, end: float, text: str, *, words: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        avg_logprob=-0.2,
        compression_ratio=1.1,
        no_speech_prob=0.01,
        words=[_word(text.strip(), start, end)] if words else [],
    )


class FakeWhisper:
    """Returns canned segments per input and records every transcribe() call."""

    def __init__(self, name: str, device: str, compute_type: str) -> None:
        self.name = name
        self.device = device
        self.compute_type = compute_type
        self.calls: list[dict[str, Any]] = []
        # Keyed by the channel array's first sample, or by "path" for a file path.
        self.script: dict[Any, list[SimpleNamespace]] = {}

    def transcribe(self, audio: Any, **options: Any):
        key = "path" if isinstance(audio, str) else float(audio[0])
        self.calls.append({"audio": audio, "options": options})
        info = SimpleNamespace(duration=10.0, language=options.get("language") or "ru")
        return iter(self.script.get(key, [])), info


@pytest.fixture
def whisper(monkeypatch) -> FakeWhisper:
    fake = FakeWhisper("fake", "cpu", "int8")
    monkeypatch.setattr(server, "_build_model", lambda *_: fake)
    monkeypatch.setattr(server.settings, "model_unload_seconds", 0.0)
    return fake


def _channel(marker: float) -> np.ndarray:
    return np.full(160, marker, dtype=np.float32)


def _stub_audio(monkeypatch, *, duration: float | None = 10.0, channels: int | None = 2) -> None:
    monkeypatch.setattr(server, "probe_audio", lambda _path: AudioInfo(duration=duration, channels=channels))


def _post(client: TestClient, **data: Any):
    return client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json", "language": "ru", **data},
        files={"file": ("call.wav", b"audio")},
    )


def test_stereo_split_tags_channels_and_sorts_by_start(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    monkeypatch.setattr(server, "decode_channels", lambda _path: [_channel(1.0), _channel(2.0)])
    whisper.script = {
        1.0: [_segment(0.0, 2.0, " Hello"), _segment(6.0, 7.0, " Bye")],
        2.0: [_segment(2.5, 4.0, " Hi there"), _segment(0.0, 1.0, " Overlap")],
    }

    with TestClient(server.app) as client:
        body = _post(client, channels="split").json()

    assert body["channel_mode"] == "split"
    assert body["channels"] == 2
    assert body["model"] == server.model_name
    assert [(s["start"], s["channel"]) for s in body["segments"]] == [(0.0, 0), (0.0, 1), (2.5, 1), (6.0, 0)]
    assert [s["id"] for s in body["segments"]] == [0, 1, 2, 3]
    assert body["text"] == " Hello Overlap Hi there Bye"
    assert len(whisper.calls) == 2
    # Every channel is a separate decode of its own samples.
    assert [float(call["audio"][0]) for call in whisper.calls] == [1.0, 2.0]


def test_split_pins_language_detected_on_first_channel(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    monkeypatch.setattr(server, "decode_channels", lambda _path: [_channel(1.0), _channel(2.0)])

    with TestClient(server.app) as client:
        _post(client, channels="split", language="")

    assert [call["options"]["language"] for call in whisper.calls] == [None, "ru"]


def test_mono_file_with_split_is_one_channel_zero(whisper, monkeypatch):
    _stub_audio(monkeypatch, channels=1)
    monkeypatch.setattr(server, "decode_channels", lambda _path: [_channel(1.0)])
    whisper.script = {1.0: [_segment(0.0, 1.0, " Solo")]}

    with TestClient(server.app) as client:
        response = _post(client, channels="split")

    body = response.json()
    assert response.status_code == 200
    assert body["channels"] == 1
    assert body["channel_mode"] == "split"
    assert [s["channel"] for s in body["segments"]] == [0]


def test_more_than_two_channels_are_each_transcribed(whisper, monkeypatch):
    _stub_audio(monkeypatch, channels=3)
    monkeypatch.setattr(server, "decode_channels", lambda _path: [_channel(1.0), _channel(2.0), _channel(3.0)])
    whisper.script = {
        1.0: [_segment(3.0, 4.0, " a")],
        2.0: [_segment(1.0, 2.0, " b")],
        3.0: [_segment(2.0, 3.0, " c")],
    }

    with TestClient(server.app) as client:
        body = _post(client, channels="split").json()

    assert body["channels"] == 3
    assert [(s["channel"], s["text"]) for s in body["segments"]] == [(1, " b"), (2, " c"), (0, " a")]


def test_mix_is_default_and_hands_the_file_path_to_the_model(whisper, monkeypatch):
    _stub_audio(monkeypatch, channels=2)
    monkeypatch.setattr(server, "decode_channels", lambda _path: pytest.fail("mix must not split channels"))
    whisper.script = {"path": [_segment(0.0, 1.0, " Mixed")]}

    with TestClient(server.app) as client:
        body = _post(client).json()

    assert body["channel_mode"] == "mix"
    assert body["channels"] == 2
    assert "channel" not in body["segments"][0]
    assert isinstance(whisper.calls[0]["audio"], str)


def test_invalid_channels_value_falls_back_to_mix(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    whisper.script = {"path": [_segment(0.0, 1.0, " Mixed")]}

    with TestClient(server.app) as client:
        body = _post(client, channels="left").json()

    assert body["channel_mode"] == "mix"


def test_word_granularity_adds_words_to_segments(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    whisper.script = {"path": [_segment(0.0, 1.0, " Hello")]}

    with TestClient(server.app) as client:
        legacy = _post(client).json()
        with_words = _post(client, **{"timestamp_granularities[]": "word"}).json()
        segment_only = _post(client, **{"timestamp_granularities[]": "segment"}).json()

    # Without granularities verbose_json keeps its historical shape.
    assert "words" not in legacy["segments"][0]
    assert legacy["words"][0]["word"] == "Hello"
    assert with_words["segments"][0]["words"] == [{"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.9}]
    assert with_words["words"][0]["word"] == "Hello"
    assert "words" not in segment_only["segments"][0]


def test_split_segments_carry_words(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    monkeypatch.setattr(server, "decode_channels", lambda _path: [_channel(1.0), _channel(2.0)])
    whisper.script = {1.0: [_segment(0.0, 1.0, " Hello")], 2.0: [_segment(1.0, 2.0, " Hi")]}

    with TestClient(server.app) as client:
        body = _post(client, channels="split", **{"timestamp_granularities[]": "word"}).json()

    assert [s["words"][0]["word"] for s in body["segments"]] == ["Hello", "Hi"]


def test_hotwords_are_normalised_and_passed_to_the_model(whisper, monkeypatch):
    _stub_audio(monkeypatch)

    with TestClient(server.app) as client:
        _post(client, hotwords="Ivanov,  Petrov\nlaser dentistry,, \n")
        _post(client, hotwords=" , \n")
        _post(client)

    assert whisper.calls[0]["options"]["hotwords"] == "Ivanov, Petrov, laser dentistry"
    assert "hotwords" not in whisper.calls[1]["options"]
    assert "hotwords" not in whisper.calls[2]["options"]


def test_hotwords_are_dropped_when_the_library_lacks_them(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    monkeypatch.setattr(server, "HOTWORDS_SUPPORTED", False)

    with TestClient(server.app) as client:
        assert _post(client, hotwords="Ivanov").status_code == 200

    assert "hotwords" not in whisper.calls[0]["options"]


def test_installed_faster_whisper_supports_hotwords():
    assert server.HOTWORDS_SUPPORTED


def test_prompt_and_hotwords_are_independent(whisper, monkeypatch):
    _stub_audio(monkeypatch)

    with TestClient(server.app) as client:
        _post(client, prompt="Clinic terms", hotwords="Ivanov")

    options = whisper.calls[0]["options"]
    assert options["initial_prompt"] == "Clinic terms"
    assert options["hotwords"] == "Ivanov"


def test_transcribe_endpoint_response_is_backward_compatible(whisper, monkeypatch):
    _stub_audio(monkeypatch)
    whisper.script = {"path": [_segment(0.0, 1.0, " Привет")]}

    with TestClient(server.app) as client:
        plain = client.post("/transcribe", params={"language": "ru"}, files={"file": ("voice.ogg", b"audio")}).json()
        words = client.post(
            "/transcribe", params={"word_timestamps": "true"}, files={"file": ("voice.ogg", b"audio")}
        ).json()

    assert plain["text"] == " Привет"
    assert plain["segments"] == 1
    assert plain["language"] == "ru"
    assert plain["task"] == "transcribe"
    assert "channel" not in plain["segment_details"][0]
    assert "words" not in plain["segment_details"][0]
    assert "words" not in plain
    assert words["words"][0]["word"] == "Привет"


def test_oversized_upload_is_rejected_with_413_before_the_body_is_read(monkeypatch):
    monkeypatch.setattr(server.settings, "max_upload_mb", 1.0)
    reached: list[bool] = []

    async def fake_enqueue(**_kwargs):
        reached.append(True)
        return {}

    monkeypatch.setattr(server, "_enqueue_transcription", fake_enqueue)
    client = TestClient(server.app)
    big = b"\0" * (3 * 1024 * 1024)

    for path in ("/transcribe", "/v1/audio/transcriptions"):
        response = client.post(path, files={"file": ("call.wav", big)})
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]
    assert not reached


def test_save_upload_streams_to_disk_and_enforces_the_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(server.settings, "max_upload_mb", 1.0)
    monkeypatch.setattr(server.tempfile, "tempdir", str(tmp_path))

    async def run(size: int) -> str:
        return await server._save_upload(UploadFile(io.BytesIO(b"\0" * size), filename="call.mp3"))

    path = asyncio.run(run(1024 * 1024))
    assert path.endswith(".mp3")
    assert len(open(path, "rb").read()) == 1024 * 1024

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(run(1024 * 1024 + 1))
    assert exc_info.value.status_code == 413
    # The partial file of the rejected upload is removed.
    assert len(list(tmp_path.glob("*.mp3"))) == 1


def test_audio_longer_than_the_limit_is_rejected_with_413(whisper, monkeypatch, tmp_path):
    monkeypatch.setattr(server.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(server.settings, "max_audio_seconds", 7500.0)
    _stub_audio(monkeypatch, duration=7501.0)

    with TestClient(server.app) as client:
        response = _post(client)

    assert response.status_code == 413
    assert "too long" in response.json()["detail"]
    assert not whisper.calls
    assert not list(tmp_path.glob("*.wav"))


def test_two_hour_audio_is_accepted(whisper, monkeypatch):
    _stub_audio(monkeypatch, duration=7200.0)
    whisper.script = {"path": [_segment(0.0, 1.0, " ok")]}

    with TestClient(server.app) as client:
        assert _post(client).status_code == 200


def test_wait_timeout_grows_with_duration(monkeypatch):
    monkeypatch.setattr(server.settings, "default_timeout_seconds", 180.0)
    monkeypatch.setattr(server.settings, "timeout_per_audio_second", 1.0)

    assert server._wait_timeout(None) == 180.0
    assert server._wait_timeout(30.0) == 180.0
    assert server._wait_timeout(7200.0) == 7200.0
    monkeypatch.setattr(server.settings, "timeout_per_audio_second", 0.5)
    assert server._wait_timeout(7200.0) == 3600.0


def test_request_wait_uses_duration_based_timeout(whisper, monkeypatch):
    waits: list[float] = []
    real_wait_for = asyncio.wait_for

    async def spy(awaitable, timeout):
        waits.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(server.asyncio, "wait_for", spy)
    monkeypatch.setattr(server.settings, "default_timeout_seconds", 180.0)
    monkeypatch.setattr(server.settings, "timeout_per_audio_second", 1.0)

    with TestClient(server.app) as client:
        _stub_audio(monkeypatch, duration=3600.0)
        _post(client)
        _stub_audio(monkeypatch, duration=20.0)
        _post(client)
        _stub_audio(monkeypatch, duration=3600.0)
        _post(client, timeout_seconds="42")

    assert waits == [3600.0, 180.0, 42.0]


def test_unprobeable_audio_uses_default_timeout(whisper, monkeypatch):
    def broken_probe(_path: str):
        raise RuntimeError("invalid data")

    monkeypatch.setattr(server, "probe_audio", broken_probe)
    assert asyncio.run(server._inspect_audio("x.wav", filename="x.wav")) == AudioInfo(None, None)
