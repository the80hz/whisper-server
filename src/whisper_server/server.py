"""FastAPI application wrapping faster-whisper transcription."""

from __future__ import annotations

import asyncio
import array
import contextlib
import logging
import os
import secrets
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
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
ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


@dataclass(slots=True)
class TranscriptionJob:
    filename: str
    audio_path: str
    task: Literal["transcribe", "translate"]
    language: str | None
    word_timestamps: bool
    initial_prompt: str | None
    temperature: float
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
    # Ensure model is loaded for the probe
    _load_model_sync()
    segments_iter, info = model.transcribe(str(HEALTH_CLIP_PATH), task="transcribe", temperature=0.0)
    text = "".join(segment.text for segment in segments_iter)
    elapsed = time.monotonic() - start
    return {
        "probe_duration": round(info.duration, 3),
        "probe_processing_seconds": round(elapsed, 3),
        "probe_text": text,
    }


def _transcribe_file(job: TranscriptionJob) -> dict[str, Any]:
    # Ensure model is loaded when running in the threadpool
    _load_model_sync()
    _update_model_last_used_sync()
    started = time.monotonic()
    segments_iter, info = model.transcribe(
        job.audio_path,
        task=job.task,
        language=job.language,
        word_timestamps=job.word_timestamps,
        initial_prompt=job.initial_prompt,
        temperature=job.temperature,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    logger.info(
        "Transcription started for %s: duration=%.2fs task=%s language=%s",
        job.filename,
        duration,
        job.task,
        job.language or "auto",
    )

    segments = []
    next_progress_percent = 10
    last_progress_log = started
    for segment in segments_iter:
        segments.append(segment)
        _update_model_last_used_sync()

        segment_end = float(getattr(segment, "end", 0.0) or 0.0)
        elapsed = time.monotonic() - started
        progress_percent = min(100.0, (segment_end / duration * 100.0) if duration > 0 else 0.0)
        should_log_percent = duration > 0 and progress_percent >= next_progress_percent
        should_log_interval = elapsed - last_progress_log >= 30.0
        if should_log_percent or should_log_interval:
            logger.info(
                "Transcription progress for %s: %.1f%% audio=%.2fs/%.2fs segments=%d elapsed=%.2fs",
                job.filename,
                progress_percent,
                segment_end,
                duration,
                len(segments),
                elapsed,
            )
            last_progress_log = time.monotonic()
            while next_progress_percent <= progress_percent:
                next_progress_percent += 10

    text = "".join(segment.text for segment in segments)
    segment_details = [
        {
            "id": index,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "avg_logprob": getattr(segment, "avg_logprob", None),
            "compression_ratio": getattr(segment, "compression_ratio", None),
            "no_speech_prob": getattr(segment, "no_speech_prob", None),
        }
        for index, segment in enumerate(segments)
    ]

    payload: dict[str, Any] = {
        "text": text,
        "duration": info.duration,
        "segments": len(segments),
        "segment_details": segment_details,
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


def _format_timestamp(seconds: float, *, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp(float(segment["start"]), separator=",")
        end = _format_timestamp(float(segment["end"]), separator=",")
        text = str(segment["text"]).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _render_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        start = _format_timestamp(float(segment["start"]), separator=".")
        end = _format_timestamp(float(segment["end"]), separator=".")
        text = str(segment["text"]).strip()
        blocks.append(f"{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def _openai_payload(result: dict[str, Any], response_format: ResponseFormat) -> dict[str, Any] | str:
    if response_format == "text":
        return str(result["text"])
    if response_format == "srt":
        return _render_srt(result["segment_details"])
    if response_format == "vtt":
        return _render_vtt(result["segment_details"])
    if response_format == "verbose_json":
        payload: dict[str, Any] = {
            "task": result["task"],
            "language": result["language"],
            "duration": result["duration"],
            "text": result["text"],
            "segments": result["segment_details"],
        }
        if "words" in result:
            payload["words"] = result["words"]
        return payload
    return {"text": result["text"]}


def _media_type(response_format: ResponseFormat) -> str:
    if response_format == "json" or response_format == "verbose_json":
        return "application/json"
    if response_format == "srt":
        return "application/x-subrip; charset=utf-8"
    if response_format == "vtt":
        return "text/vtt; charset=utf-8"
    return "text/plain; charset=utf-8"


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, settings.api_token):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _save_upload(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    suffix = Path(file.filename).suffix or ".tmp"
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
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
    return audio_path


async def _enqueue_transcription(
    *,
    file: UploadFile,
    task: Literal["transcribe", "translate"],
    language: str | None,
    word_timestamps: bool,
    timeout_seconds: float | None,
    initial_prompt: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    logger.info("Received file %s", file.filename)
    audio_path = await _save_upload(file)

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    job = TranscriptionJob(
        filename=file.filename,
        audio_path=audio_path,
        task=task,
        language=language,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        temperature=temperature,
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
    # Load model at startup to preserve previous eager-loading behaviour
    try:
        await run_in_threadpool(_load_model_sync)
    except Exception:
        logger.exception("Failed to load Whisper model at startup")

    # Start idle watcher if enabled
    global model_watcher_task
    if settings.model_unload_seconds and settings.model_unload_seconds > 0:
        model_watcher_task = asyncio.create_task(_model_idle_watcher())


@app.on_event("shutdown")
async def _shutdown_worker() -> None:
    if worker_task is None:
        return
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    # Stop model watcher and unload model
    global model_watcher_task
    if model_watcher_task is not None:
        model_watcher_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await model_watcher_task
    try:
        await run_in_threadpool(_unload_model_sync)
    except Exception:
        logger.exception("Failed to unload Whisper model at shutdown")

# Model instance and idle-unload management.
# The model is loaded at startup (to keep previous behaviour) and may be
# automatically unloaded after `settings.model_unload_seconds` of idle time.
model: WhisperModel | None = None
model_last_used: float = 0.0
model_watcher_task: asyncio.Task[None] | None = None


def _update_model_last_used_sync() -> None:
    global model_last_used
    model_last_used = time.monotonic()


def _load_model_sync() -> None:
    """Load the Whisper model synchronously (safe to call from a thread).

    Idempotent: will not reload if already loaded.
    """
    global model
    if model is not None:
        _update_model_last_used_sync()
        return
    logger.info(
        "Loading Whisper model %s device=%s compute_type=%s",
        settings.whisper_model,
        settings.device,
        settings.compute_type,
    )
    model = WhisperModel(settings.whisper_model, device=settings.device, compute_type=settings.compute_type)
    _update_model_last_used_sync()


def _unload_model_sync() -> None:
    """Unload the Whisper model and free caches (best-effort)."""
    global model
    if model is None:
        return
    logger.info("Unloading Whisper model from memory due to idleness")
    try:
        # Remove reference to underlying implementation if present
        impl = getattr(model, "model", None)
        if impl is not None:
            try:
                del impl
            except Exception:
                pass
    except Exception:
        pass
    model = None
    # Best-effort garbage collection and CUDA cache clear
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # torch may not be installed in some environments
        pass


async def _model_idle_watcher() -> None:
    """Background task: unload model after configured idle timeout."""
    poll_interval = 5.0
    while True:
        try:
            if model is not None and settings.model_unload_seconds and settings.model_unload_seconds > 0:
                idle = time.monotonic() - model_last_used
                if idle >= settings.model_unload_seconds:
                    await run_in_threadpool(_unload_model_sync)
        except Exception:
            logger.exception("Model idle watcher encountered an error")
        await asyncio.sleep(poll_interval)


def _runtime_device() -> str:
    """Return the device actually chosen by faster-whisper."""

    if model is None:
        return settings.device
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
    _: None = Depends(_check_auth),
    file: UploadFile = File(...),
    task: Literal["transcribe", "translate"] = Query(default="transcribe"),
    language: str | None = Query(default=None),
    word_timestamps: bool = Query(default=False),
    timeout_seconds: float | None = Query(default=None, gt=0),
) -> dict[str, Any]:
    return await _enqueue_transcription(
        file=file,
        task=task,
        language=language,
        word_timestamps=word_timestamps,
        timeout_seconds=timeout_seconds,
    )


@app.post("/v1/audio/transcriptions", response_model=None)
async def openai_audio_transcriptions(
    _: None = Depends(_check_auth),
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: ResponseFormat = Form(default="json"),
    temperature: float = Form(default=0.0),
    timeout_seconds: float | None = Form(default=None),
) -> Any:
    if model not in {"whisper-1", settings.whisper_model}:
        logger.info("Ignoring OpenAI-compatible model=%s; using configured model=%s", model, settings.whisper_model)

    word_timestamps = response_format == "verbose_json"
    result = await _enqueue_transcription(
        file=file,
        task="transcribe",
        language=language,
        word_timestamps=word_timestamps,
        timeout_seconds=timeout_seconds,
        initial_prompt=prompt,
        temperature=temperature,
    )
    payload = _openai_payload(result, response_format)
    if isinstance(payload, str):
        return Response(content=payload, media_type=_media_type(response_format))
    return payload


@app.post("/v1/audio/translations", response_model=None)
async def openai_audio_translations(
    _: None = Depends(_check_auth),
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    prompt: str | None = Form(default=None),
    response_format: ResponseFormat = Form(default="json"),
    temperature: float = Form(default=0.0),
    timeout_seconds: float | None = Form(default=None),
) -> Any:
    if model not in {"whisper-1", settings.whisper_model}:
        logger.info("Ignoring OpenAI-compatible model=%s; using configured model=%s", model, settings.whisper_model)

    word_timestamps = response_format == "verbose_json"
    result = await _enqueue_transcription(
        file=file,
        task="translate",
        language=None,
        word_timestamps=word_timestamps,
        timeout_seconds=timeout_seconds,
        initial_prompt=prompt,
        temperature=temperature,
    )
    payload = _openai_payload(result, response_format)
    if isinstance(payload, str):
        return Response(content=payload, media_type=_media_type(response_format))
    return payload


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("whisper_server.server:app", host="0.0.0.0", port=settings.port)
