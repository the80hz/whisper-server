"""FastAPI application wrapping faster-whisper transcription."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

from .config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whisper-api")

app = FastAPI(title="Whisper Server", version="0.1.0")

# Instantiate the Whisper model once at module import for best latency/memory trade-offs.
model = WhisperModel(
    settings.whisper_model,
    device=settings.device,
    compute_type=settings.compute_type,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.whisper_model}


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
