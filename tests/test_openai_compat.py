from fastapi import HTTPException
import pytest

from whisper_server import server


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

    assert server._check_auth(None) is None


def test_auth_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", "secret")

    with pytest.raises(HTTPException) as exc_info:
        server._check_auth("Bearer wrong")

    assert exc_info.value.status_code == 401


def test_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setattr(server.settings, "api_token", "secret")

    assert server._check_auth("Bearer secret") is None
