from typing import Any, cast

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
import pytest

from whisper_server import server
from whisper_server.config import Settings


def _result():
    return {
        "task": "transcribe",
        "language": "ru",
        "duration": 2.5,
        "text": " Привет мир",
        "segment_details": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " Привет мир",
            }
        ],
        "words": [
            {
                "word": "Привет",
                "start": 0.0,
                "end": 1.0,
                "probability": 0.9,
            }
        ],
    }


def test_openai_json_payload_returns_text_object():
    assert server._openai_payload(_result(), "json") == {"text": " Привет мир"}


def test_openai_verbose_json_payload_includes_segments_and_words():
    payload = server._openai_payload(_result(), "verbose_json")

    assert payload["text"] == " Привет мир"
    assert payload["segments"][0]["start"] == 0.0
    assert payload["words"][0]["word"] == "Привет"


def test_srt_rendering():
    payload = server._openai_payload(_result(), "srt")

    assert "1\n00:00:00,000 --> 00:00:02,500\nПривет мир" in payload


def test_vtt_rendering():
    payload = server._openai_payload(_result(), "vtt")

    assert payload.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500")


def test_auth_is_noop_without_token(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", None)
    monkeypatch.setattr(server.settings, "api_key", None)

    assert server._check_auth(None) is None


def test_auth_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", "secret")
    monkeypatch.setattr(server.settings, "api_key", None)

    with pytest.raises(HTTPException) as exc_info:
        server._check_auth("Bearer wrong")

    assert exc_info.value.status_code == 401


def test_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", "secret")
    monkeypatch.setattr(server.settings, "api_key", None)

    assert server._check_auth("Bearer secret") is None


def test_auth_accepts_legacy_api_key(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", None)
    monkeypatch.setattr(server.settings, "api_key", "legacy-secret")

    assert server._check_auth("Bearer legacy-secret") is None


def _job(*, temperature: float = 0.0, word_timestamps: bool = False):
    return server.TranscriptionJob(
        filename="sample.ogg",
        audio_path="/tmp/sample.ogg",
        task="transcribe",
        language=None,
        word_timestamps=word_timestamps,
        initial_prompt=None,
        temperature=temperature,
        future=cast(Any, None),
    )


def test_transcription_options_enable_repetition_safeguards(monkeypatch):
    monkeypatch.setattr(server.settings, "temperature_fallback", True)
    monkeypatch.setattr(server.settings, "hallucination_silence_threshold", 1.0)
    monkeypatch.setattr(server.settings, "vad_filter", True)

    options = server._transcription_options(_job())

    assert options["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert options["condition_on_previous_text"] is False
    assert options["repetition_penalty"] == 1.1
    assert options["no_repeat_ngram_size"] == 3
    assert options["word_timestamps"] is True
    assert options["hallucination_silence_threshold"] == 1.0
    assert options["vad_parameters"]["min_silence_duration_ms"] == 500


def test_explicit_nonzero_temperature_disables_fallback(monkeypatch):
    monkeypatch.setattr(server.settings, "temperature_fallback", True)

    options = server._transcription_options(_job(temperature=0.4))

    assert options["temperature"] == 0.4


@pytest.mark.parametrize(
    ("adapter", "value", "default"),
    [
        (server.TaskArgument, "invalid", "transcribe"),
        (server.ResponseFormatArgument, "invalid", "json"),
        (server.LanguageArgument, "invalid", None),
        (server.BoolArgument, "invalid", False),
        (server.TemperatureArgument, "nan", 0.0),
        (server.TemperatureArgument, "2", 0.0),
        (server.TimeoutArgument, "invalid", None),
        (server.TimeoutArgument, "-1", None),
    ],
)
def test_invalid_request_arguments_use_defaults(adapter, value, default):
    assert TypeAdapter(adapter).validate_python(value) == default


def test_invalid_decode_settings_use_defaults():
    invalid = Settings(
        _env_file=None,
        device="tpu",
        compute_type="quantum",
        vad_filter="maybe",
        vad_threshold="nan",
        repetition_penalty="0.5",
        no_repeat_ngram_size="-3",
        hallucination_silence_threshold="-1",
    )

    assert invalid.device == "auto"
    assert invalid.compute_type == "int8"
    assert invalid.vad_filter is True
    assert invalid.vad_threshold == 0.5
    assert invalid.repetition_penalty == 1.1
    assert invalid.no_repeat_ngram_size == 3
    assert invalid.hallucination_silence_threshold == 1.0


def test_invalid_http_arguments_reach_handler_as_defaults(monkeypatch):
    received = {}

    async def fake_enqueue(**kwargs):
        received.update(kwargs)
        return _result()

    monkeypatch.setattr(server, "_enqueue_transcription", fake_enqueue)
    client = TestClient(server.app)
    response = client.post(
        "/transcribe",
        params={
            "task": "invalid",
            "language": "invalid",
            "word_timestamps": "invalid",
            "timeout_seconds": "invalid",
        },
        files={"file": ("sample.ogg", b"audio")},
    )

    assert response.status_code == 200
    assert received["task"] == "transcribe"
    assert received["language"] is None
    assert received["word_timestamps"] is False
    assert received["timeout_seconds"] is None


def test_invalid_openai_form_arguments_reach_handler_as_defaults(monkeypatch):
    received = {}

    async def fake_enqueue(**kwargs):
        received.update(kwargs)
        return _result()

    monkeypatch.setattr(server, "_enqueue_transcription", fake_enqueue)
    client = TestClient(server.app)
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "language": "invalid",
            "response_format": "invalid",
            "temperature": "nan",
            "timeout_seconds": "invalid",
        },
        files={"file": ("sample.ogg", b"audio")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": " Привет мир"}
    assert received["language"] is None
    assert received["temperature"] == 0.0
    assert received["timeout_seconds"] is None


def test_api_token_takes_precedence_over_legacy_api_key(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", "new-secret")
    monkeypatch.setattr(server.settings, "api_key", "legacy-secret")

    assert server._check_auth("Bearer new-secret") is None
