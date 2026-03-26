"""FastAPI application wrapping faster-whisper transcription."""

from __future__ import annotations

import asyncio
import array
import contextlib
import logging
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
transcription_queue: asyncio.Queue["TranscriptionJob"] = asyncio.Queue(maxsize=settings.queue_max_size)
worker_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class TranscriptionJob:
    filename: str
    audio_path: str
    task: Literal["transcribe", "translate"]
    language: str | None
    word_timestamps: bool
    future: asyncio.Future[dict[str, Any]]


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


def _transcribe_file(job: TranscriptionJob) -> dict[str, Any]:
    segments_iter, info = model.transcribe(
        job.audio_path,
        task=job.task,
        language=job.language,
        word_timestamps=job.word_timestamps,
    )
    segments = list(segments_iter)
    text = "".join(segment.text for segment in segments)

    payload: dict[str, Any] = {
        "text": text,
        "duration": info.duration,
        "segments": len(segments),
        "language": getattr(info, "language", None),
        "task": job.task,
    }
    if job.word_timestamps:
        payload["words"] = [
            {
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for segment in segments
            for word in (segment.words or [])
        ]
    return payload


async def _transcription_worker() -> None:
    while True:
        job = await transcription_queue.get()
        started = time.monotonic()
        try:
            result = await run_in_threadpool(_transcribe_file, job)
            result["processing_seconds"] = time.monotonic() - started
            if not job.future.done():
                job.future.set_result(result)
            logger.info(
                "Transcribed %s: duration=%.2fs segments=%d processing=%.2fs queue=%d",
                job.filename,
                result["duration"],
                result["segments"],
                result["processing_seconds"],
                transcription_queue.qsize(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcription failed for %s", job.filename)
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(job.audio_path)
            transcription_queue.task_done()


@app.on_event("startup")
async def _startup_worker() -> None:
    global worker_task
    worker_task = asyncio.create_task(_transcription_worker())


@app.on_event("shutdown")
async def _shutdown_worker() -> None:
    if worker_task is None:
        return
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task

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


def _format_uptime(seconds: float) -> str:
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if int(days):
        parts.append(f"{int(days)}d")
    parts.append(f"{int(hours):02}:{int(minutes):02}:{secs:05.2f}")
    return " ".join(parts)


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

    uptime_seconds = now - app_started_at
    return {
        "status": status,
        "model": settings.whisper_model,
        "device": _runtime_device(),
        "compute_type": settings.compute_type,
        "log_level": settings.log_level.upper(),
        "queue_size": str(transcription_queue.qsize()),
        "queue_capacity": str(settings.queue_max_size),
        "uptime": _format_uptime(uptime_seconds),
        "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        **probe,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    task: Literal["transcribe", "translate"] = Query(default="transcribe"),
    language: str | None = Query(default=None),
    word_timestamps: bool = Query(default=False),
    timeout_seconds: float | None = Query(default=None, gt=0),
    max_file_size_mb: float | None = Query(default=None, gt=0),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    logger.info("Received file %s", file.filename)

    suffix = Path(file.filename).suffix or ".tmp"
    max_bytes = int((max_file_size_mb or settings.max_upload_mb) * 1024 * 1024)
    total_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        audio_path = tmp_file.name
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Limit is {(max_bytes / 1024 / 1024):.2f} MB",
                    )
                tmp_file.write(chunk)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(audio_path)
            raise

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    job = TranscriptionJob(
        filename=file.filename,
        audio_path=audio_path,
        task=task,
        language=language,
        word_timestamps=word_timestamps,
        future=future,
    )
    try:
        transcription_queue.put_nowait(job)
    except asyncio.QueueFull:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(audio_path)
        raise HTTPException(status_code=429, detail="Transcription queue is full. Try again later.") from None

    wait_timeout = timeout_seconds or settings.default_timeout_seconds
    try:
        result = await asyncio.wait_for(asyncio.shield(future), timeout=wait_timeout)
        result["queue_position_left"] = transcription_queue.qsize()
        return result
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Transcription did not finish within {wait_timeout:.1f}s",
        ) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("whisper_server.server:app", host="0.0.0.0", port=settings.port)
