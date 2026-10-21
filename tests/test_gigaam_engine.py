"""Tests for the GigaAM engine: chunk planning, phrases, batching and the server switch.

The model itself is not loaded: these parts are plain Python.
"""

from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest

from whisper_server import server
from whisper_server.audio import AudioInfo
from whisper_server.chunking import plan_chunks
from whisper_server.gigaam_engine import Info, _drop_flash_import_error, batches, levelled, phrases_of_chunk, unheard

BUILD_MODEL = server._build_model


def test_speech_shorter_than_a_chunk_is_one_padded_chunk():
    assert plan_chunks([(1.0, 5.0)], duration=10.0) == [(0.7, 5.3)]


def test_padding_takes_at_most_half_of_the_gap_to_the_edge():
    assert plan_chunks([(0.1, 9.95)], duration=10.0) == [(0.05, 9.975)]


def test_long_speech_is_cut_at_the_longest_late_pause():
    speech = [(0.0, 8.0), (8.2, 14.0), (16.0, 20.0), (20.1, 30.0)]

    chunks = plan_chunks(speech, duration=30.0, max_chunk=24.0)

    # The 2 s pause after 14.0 wins over the later 0.1 s one.
    assert chunks == [(0.0, 14.3), (15.7, 30.0)]
    assert all(end - start <= 24.0 for start, end in chunks)


def test_speech_without_pauses_is_cut_into_equal_pieces():
    chunks = plan_chunks([(0.0, 60.0)], duration=60.0, max_chunk=24.0)

    assert len(chunks) == 3
    assert all(end - start <= 24.0 for start, end in chunks)


def test_a_chunk_becomes_phrases_at_pauses_and_times_move_to_the_recording():
    words = [("Алло.", 0.0, 0.4), ("Здравствуйте,", 0.7, 1.3), ("слушаю", 1.4, 1.8), ("Да", 2.6, 2.8)]

    segments = phrases_of_chunk(words, offset=10.0)

    # 0.3 s after a full stop ends the phrase; 0.8 s ends it anywhere.
    assert [s.text for s in segments] == [" Алло.", " Здравствуйте, слушаю", " Да"]
    assert (segments[1].start, segments[1].end) == (10.7, 11.8)
    assert segments[1].words[0].word == " Здравствуйте,"
    assert segments[1].words[0].probability is None


def test_a_dialogue_dash_is_not_a_word():
    segments = phrases_of_chunk([("—", 0.0, 0.1), ("Да.", 0.1, 0.4)], offset=0.0)

    assert [w.word for s in segments for w in s.words] == [" Да."]


def test_a_chunk_of_dashes_gives_no_segment():
    assert phrases_of_chunk([("—", 0.0, 0.1)], offset=0.0) == []


def test_batches_put_chunks_of_similar_length_together():
    chunks = [(0.0, 2.0), (3.0, 23.0), (24.0, 25.0), (26.0, 45.0)]

    assert batches(chunks, 2) == [[(3.0, 23.0), (26.0, 45.0)], [(0.0, 2.0), (24.0, 25.0)]]


def test_gigaam_engine_is_built_from_its_own_settings(monkeypatch):
    built: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        "whisper_server.gigaam_engine.GigaAmModel", lambda *args, **kwargs: built.append((args, kwargs))
    )
    monkeypatch.setattr(server.settings, "engine", "gigaam")
    monkeypatch.setattr(server.settings, "gigaam_batch_size", 8)
    monkeypatch.setattr(server.settings, "gigaam_download_root", "/models")

    BUILD_MODEL("v3_e2e_rnnt", "cuda", "int8")

    assert built == [(("v3_e2e_rnnt", "cuda"), {"download_root": "/models", "batch_size": 8, "quiet_speech": True})]


def test_gigaam_engine_serves_the_same_endpoint(monkeypatch):
    class FakeGigaAm:
        def transcribe(self, _audio: Any, **options: Any):
            assert options["hotwords"] == "Иванов"  # accepted, and ignored by the real engine
            return iter(phrases_of_chunk([("Добрый", 0.0, 0.3), ("день.", 0.35, 0.6)], offset=1.0)), Info(duration=5.0)

    monkeypatch.setattr(server.settings, "engine", "gigaam")
    monkeypatch.setattr(server.settings, "gigaam_model", "v3_e2e_rnnt")
    monkeypatch.setattr(server.settings, "model_unload_seconds", 0.0)
    monkeypatch.setattr(server, "_build_model", lambda *_: FakeGigaAm())
    monkeypatch.setattr(server, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(server, "probe_audio", lambda _path: AudioInfo(duration=5.0, channels=1))

    with TestClient(server.app) as client:
        body = client.post(
            "/v1/audio/transcriptions",
            data={"response_format": "verbose_json", "timestamp_granularities[]": "word", "hotwords": "Иванов"},
            files={"file": ("call.mp3", b"audio")},
        ).json()
        health = client.get("/health").json()

    assert body["text"] == " Добрый день."
    assert body["model"] == "v3_e2e_rnnt"
    assert body["language"] == "ru"
    assert body["segments"][0]["words"] == [
        {"word": " Добрый", "start": 1.0, "end": 1.3, "probability": None},
        {"word": " день.", "start": 1.35, "end": 1.6, "probability": None},
    ]
    assert health["engine"] == "gigaam"


def test_unknown_engine_falls_back_to_whisper():
    assert server.settings.__class__(engine="vosk").engine == "whisper"


def test_gigaam_without_its_packages_says_what_to_install(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_gigaam(name: str, *args: Any, **kwargs: Any):
        if name == "gigaam":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_gigaam)
    from whisper_server.gigaam_engine import GigaAmModel

    with pytest.raises(RuntimeError, match="--extra gigaam"):
        GigaAmModel("v3_e2e_rnnt", "cpu")


def test_the_flash_attn_import_error_does_not_pin_the_frames_that_imported_it(monkeypatch):
    import sys
    import types

    try:
        raise ImportError("No module named 'flash_attn'")
    except ImportError as error:
        stored = error
    assert stored.__traceback__ is not None
    encoder = types.ModuleType("gigaam.encoder")
    encoder.IMPORT_FLASH_ERR = stored  # type: ignore[attr-defined]
    package = types.ModuleType("gigaam")
    package.encoder = encoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaam", package)
    monkeypatch.setitem(sys.modules, "gigaam.encoder", encoder)

    _drop_flash_import_error()

    assert stored.__traceback__ is None


def test_levelling_makes_a_quiet_voice_as_loud_as_a_loud_one():
    rate = 16_000
    time = np.arange(rate * 2) / rate
    loud = 0.5 * np.sin(2 * np.pi * 220 * time)
    quiet = 0.01 * np.sin(2 * np.pi * 220 * time)
    silence = np.zeros(rate * 2)
    wave = np.concatenate([loud, quiet, silence]).astype(np.float32)

    out = levelled(wave)

    def rms(part: np.ndarray) -> float:
        return float(np.sqrt((part**2).mean()))

    # Inside each stretch, away from the borders the smoothing blurs.
    loud_out, quiet_out = out[rate // 2 : rate * 3 // 2], out[rate * 5 // 2 : rate * 7 // 2]
    assert rms(quiet_out) == pytest.approx(rms(loud_out), rel=0.05)
    assert rms(quiet) < rms(loud) / 40
    # The gain is capped: silence stays silence.
    assert rms(out[rate * 9 // 2 : rate * 11 // 2]) < 1e-4
    assert out.dtype == np.float32
    assert len(out) == len(wave)


def test_levelling_an_empty_recording_is_itself():
    assert len(levelled(np.zeros(0, dtype=np.float32))) == 0


def test_of_the_second_look_only_words_in_the_silence_of_the_first_are_kept():
    [found] = phrases_of_chunk(
        [("Центр", 0.0, 0.3), ("иммунологии.", 0.35, 0.9), ("Здравствуйте,", 1.0, 1.5), ("девушка", 1.55, 1.9)],
        offset=20.0,
    )
    heard = [(21.05, 21.5), (21.55, 21.9), (22.4, 22.9)]

    kept = unheard(found, heard)

    # The client's words are in the transcript already; the greeting before them is new.
    assert [segment.text for segment in kept] == [" Центр иммунологии."]
    assert (kept[0].start, kept[0].end) == (20.0, 20.9)
    assert unheard(found, [(19.0, 23.0)]) == []


def test_quiet_speech_setting_reaches_the_engine(monkeypatch):
    built: list[dict[str, Any]] = []
    monkeypatch.setattr("whisper_server.gigaam_engine.GigaAmModel", lambda *args, **kwargs: built.append(kwargs))
    monkeypatch.setattr(server.settings, "engine", "gigaam")
    monkeypatch.setattr(server.settings, "gigaam_quiet_speech", False)

    BUILD_MODEL("v3_e2e_rnnt", "cuda", "int8")

    assert built[0]["quiet_speech"] is False
    assert server.settings.__class__().gigaam_quiet_speech is True
