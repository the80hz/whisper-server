"""Tests for TRANSCRIBE_WORKERS: several recordings are transcribed at the same time."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from whisper_server import server
from whisper_server.audio import AudioInfo

# The shared fixture replaces `_build_model` with a stub for every test.
BUILD_MODEL = server._build_model


class WaitingWhisper:
    """Holds every transcribe() call until `expected` of them are running together."""

    def __init__(self, expected: int, timeout: float = 5.0) -> None:
        self.together = threading.Barrier(expected, timeout=timeout)

    def transcribe(self, _audio: Any, **options: Any):
        self.together.wait()
        return iter([]), SimpleNamespace(duration=1.0, language=options.get("language") or "ru")


def _post(client: TestClient) -> int:
    return client.post("/transcribe", files={"file": ("call.wav", b"audio")}).status_code


@pytest.fixture
def audio(monkeypatch) -> None:
    monkeypatch.setattr(server, "probe_audio", lambda _path: AudioInfo(duration=1.0, channels=1))
    monkeypatch.setattr(server.settings, "model_unload_seconds", 0.0)


def test_recordings_are_transcribed_together(audio, monkeypatch):
    fake = WaitingWhisper(expected=3)
    monkeypatch.setattr(server, "_build_model", lambda *_: fake)
    monkeypatch.setattr(server.settings, "transcribe_workers", 3)

    with TestClient(server.app) as client, ThreadPoolExecutor(3) as pool:
        codes = list(pool.map(lambda _: _post(client), range(3)))
        workers = client.get("/health").json()["transcribe_workers"]

    assert codes == [200, 200, 200]
    assert workers == "3"


def test_one_worker_takes_recordings_in_turn(audio, monkeypatch):
    fake = WaitingWhisper(expected=2, timeout=0.5)
    monkeypatch.setattr(server, "_build_model", lambda *_: fake)
    monkeypatch.setattr(server.settings, "transcribe_workers", 1)

    with TestClient(server.app, raise_server_exceptions=False) as client, ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(lambda _: _post(client), range(2)))

    # The two calls never meet inside the model, so each gives up waiting for the other.
    assert codes == [500, 500]


def test_the_model_is_built_for_as_many_calls_as_there_are_workers(monkeypatch):
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(server, "WhisperModel", lambda _name, **kwargs: built.append(kwargs))

    monkeypatch.setattr(server.settings, "transcribe_workers", 4)
    BUILD_MODEL("large-v3-turbo", "cuda", "int8_float16")
    monkeypatch.setattr(server.settings, "transcribe_workers", 1)
    BUILD_MODEL("large-v3-turbo", "cuda", "int8_float16")

    assert built[0]["num_workers"] == 4
    assert "num_workers" not in built[1]


def test_worker_count_below_one_means_one(monkeypatch):
    monkeypatch.setattr(server.settings, "transcribe_workers", 0)

    assert server._transcribe_workers() == 1
