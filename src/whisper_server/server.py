"""FastAPI application wrapping faster-whisper transcription."""

from __future__ import annotations

import array
import logging
import os
import tempfile
import time
import wave
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

from .config import settings

log_path = Path(settings.log_file).expanduser()
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger("whisper-api")

app = FastAPI(title="Whisper Server", version="0.1.0")
app_started_at = time.time()


def _ensure_health_clip() -> Path:
    """Ensure a tiny silent WAV exists for health probes."""

    health_path = Path(tempfile.gettempdir()) / "whisper_health.wav"
    if health_path.exists():
        return health_path

    sample_rate = 16_000
    sample_count = sample_rate // 10  # 100 ms of audio
    silence = array.array("h", [0]) * sample_count

    with wave.open(str(health_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence.tobytes())

    return health_path


HEALTH_CLIP_PATH = _ensure_health_clip()


def _run_probe() -> dict[str, float | str | int]:
    """Run a minimal transcription to verify the model end-to-end."""

    start = time.monotonic()
    segments_iter, info = model.transcribe(str(HEALTH_CLIP_PATH), task="transcribe", temperature=0.0)
    text = "".join(segment.text for segment in segments_iter)
    elapsed = time.monotonic() - start
    return {
        "probe_duration": round(info.duration, 3),
        "probe_processing_seconds": round(elapsed, 3),
        "probe_text": text,
    }

# Instantiate the Whisper model once at module import for best latency/memory trade-offs.
model = WhisperModel(
    settings.whisper_model,
    device=settings.device,
    compute_type=settings.compute_type,
)


def _runtime_device() -> str:
    """Return the device actually chosen by faster-whisper."""

    impl = getattr(model, "model", None)
    actual = getattr(impl, "device", None) or getattr(model, "device", None)
    return str(actual) if actual else settings.device


@app.get("/health")
async def health() -> dict[str, float | str]:
    now = time.time()
    try:
        probe = await run_in_threadpool(_run_probe)
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - we want the message in health output
        logger.exception("Health probe failed")
        probe = {"probe_error": str(exc)}
        status = "error"

    return {
        "status": status,
        "model": settings.whisper_model,
        "device": _runtime_device(),
        "compute_type": settings.compute_type,
        "log_level": settings.log_level.upper(),
        "uptime_seconds": round(now - app_started_at, 2),
        "timestamp": now,
        **probe,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, float | str | int]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    start_time = time.monotonic()
    logger.info("Received file %s", file.filename)

    suffix = Path(file.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        payload = await file.read()
        tmp_file.write(payload)
        audio_path = tmp_file.name

    try:
        # Run transcription end to end, buffering segments for a single response.
        segments_iter, info = model.transcribe(audio_path, task="transcribe")
        segments = list(segments_iter)
        text = "".join(segment.text for segment in segments)
    finally:
        os.unlink(audio_path)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Transcribed %s: duration=%.2fs segments=%d processing=%.2fs",
        file.filename,
        info.duration,
        len(segments),
        elapsed,
    )

    return {
        "text": text,
        "duration": info.duration,
        "segments": len(segments),
        "processing_seconds": elapsed,
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("whisper_server.server:app", host="0.0.0.0", port=settings.port)
