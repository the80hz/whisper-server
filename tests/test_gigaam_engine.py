"""Tests for the GigaAM engine: chunk planning, phrases, batching and the server switch.

The model itself is not loaded: these parts are plain Python.
"""

from typing import Any

from fastapi.testclient import TestClient
import pytest

from whisper_server import server
from whisper_server.audio import AudioInfo
from whisper_server.chunking import plan_chunks
from whisper_server.gigaam_engine import Info, batches, phrases_of_chunk

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

    assert built == [(("v3_e2e_rnnt", "cuda"), {"download_root": "/models", "batch_size": 8})]


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
