"""Shared fixtures.

The app's lifespan loads a Whisper model and binds the transcription queue to
the running event loop, so every test gets a stub model and a fresh queue.
"""

import asyncio

import pytest

from whisper_server import server


class StubModel:
    def __init__(self, name: str, device: str, compute_type: str) -> None:
        self.name = name
        self.device = device
        self.compute_type = compute_type


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch):
    monkeypatch.setattr(server, "_build_model", StubModel)
    server.transcription_queue = asyncio.Queue(maxsize=server.settings.queue_max_size)
    server.model = None
    server.model_in_use = 0
    server.model_last_used = 0.0
    server.model_cpu_fallback = False
    server.model_name = server.settings.whisper_model
    server.model_device = server.settings.device
    server.model_compute_type = server.settings.compute_type
    yield
    server.model = None
    server.model_in_use = 0
