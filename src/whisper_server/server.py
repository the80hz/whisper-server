"""FastAPI application wrapping faster-whisper transcription."""

from __future__ import annotations

import asyncio
import array
import contextlib
import logging
import os
import secrets
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES
from pydantic import BeforeValidator

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

app_started_at = time.time()
transcription_queue: asyncio.Queue["TranscriptionJob"] = asyncio.Queue(maxsize=settings.queue_max_size)
worker_task: asyncio.Task[None] | None = None
ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


def _argument_default(name: str, value: Any, default: Any) -> Any:
    logger.warning("Invalid request argument %s=%r; using default %r", name, value, default)
    return default


def _valid_task(value: Any) -> str:
    return value if value in {"transcribe", "translate"} else _argument_default("task", value, "transcribe")


def _valid_response_format(value: Any) -> str:
    valid = {"json", "text", "srt", "verbose_json", "vtt"}
    return value if value in valid else _argument_default("response_format", value, "json")


def _valid_language(value: Any) -> str | None:
    if value is None or value == "":
        return None
    normalized = str(value).lower().strip()
    return normalized if normalized in _LANGUAGE_CODES else _argument_default("language", value, None)


def _valid_bool_argument(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).lower().strip()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return _argument_default("word_timestamps", value, False)


def _valid_temperature(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return _argument_default("temperature", value, 0.0)
    return parsed if isfinite(parsed) and 0 <= parsed <= 1 else _argument_default("temperature", value, 0.0)


def _valid_timeout(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return _argument_default("timeout_seconds", value, None)
    return parsed if isfinite(parsed) and parsed > 0 else _argument_default("timeout_seconds", value, None)


TaskArgument = Annotated[Literal["transcribe", "translate"], BeforeValidator(_valid_task)]
ResponseFormatArgument = Annotated[ResponseFormat, BeforeValidator(_valid_response_format)]
LanguageArgument = Annotated[str | None, BeforeValidator(_valid_language)]
BoolArgument = Annotated[bool, BeforeValidator(_valid_bool_argument)]
TemperatureArgument = Annotated[float, BeforeValidator(_valid_temperature)]
TimeoutArgument = Annotated[float | None, BeforeValidator(_valid_timeout)]


@asynccontextmanager
async def _lifespan(_: FastAPI):
    await _startup_worker()
    try:
        yield
    finally:
        await _shutdown_worker()


app = FastAPI(title="Whisper Server", version="0.1.0", lifespan=_lifespan)


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

    def run(instance: WhisperModel) -> tuple[Any, str]:
        segments_iter, info = instance.transcribe(str(HEALTH_CLIP_PATH), task="transcribe", temperature=0.0)
        return info, "".join(segment.text for segment in segments_iter)

    info, text = _run_with_cuda_fallback(run, what="health probe")
    elapsed = time.monotonic() - start
    return {
        "probe_duration": round(info.duration, 3),
        "probe_processing_seconds": round(elapsed, 3),
        "probe_text": text,
    }


def _collect_segments(instance: WhisperModel, job: TranscriptionJob) -> tuple[list[Any], Any]:
    """Decode `job` with `instance`, logging progress as segments arrive."""

    started = time.monotonic()
    segments_iter, info = instance.transcribe(job.audio_path, **_transcription_options(job))
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

    return segments, info


def _transcribe_file(job: TranscriptionJob) -> dict[str, Any]:
    segments, info = _run_with_cuda_fallback(
        lambda instance: _collect_segments(instance, job),
        what=f"transcription of {job.filename}",
    )
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


def _transcription_options(job: TranscriptionJob) -> dict[str, Any]:
    """Build faster-whisper options with safeguards against repetition loops."""

    temperature: float | tuple[float, ...] = job.temperature
    if settings.temperature_fallback and job.temperature == 0:
        temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # faster-whisper's hallucination-on-silence detection requires word timestamps,
    # but words are only included in the API response when the caller requested them.
    detect_silence_hallucinations = settings.hallucination_silence_threshold > 0
    options: dict[str, Any] = {
        "task": job.task,
        "language": job.language,
        "word_timestamps": job.word_timestamps or detect_silence_hallucinations,
        "initial_prompt": job.initial_prompt,
        "temperature": temperature,
        "vad_filter": settings.vad_filter,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "repetition_penalty": settings.repetition_penalty,
        "no_repeat_ngram_size": settings.no_repeat_ngram_size,
        "compression_ratio_threshold": settings.compression_ratio_threshold,
        "log_prob_threshold": settings.log_prob_threshold,
        "no_speech_threshold": settings.no_speech_threshold,
    }
    if settings.vad_filter:
        options["vad_parameters"] = {
            "threshold": settings.vad_threshold,
            "min_silence_duration_ms": settings.vad_min_silence_duration_ms,
            "speech_pad_ms": settings.vad_speech_pad_ms,
        }
    if detect_silence_hallucinations:
        options["hallucination_silence_threshold"] = settings.hallucination_silence_threshold
    return options


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


def _configured_api_token() -> str | None:
    return settings.api_token or settings.api_key


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    api_token = _configured_api_token()
    if not api_token:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, api_token):
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
        await run_in_threadpool(lambda: _unload_model_sync(force=True))
    except Exception:
        logger.exception("Failed to unload Whisper model at shutdown")

# Model instance and idle-unload management.
# The model is loaded at startup (to keep previous behaviour) and may be
# automatically unloaded after `settings.model_unload_seconds` of idle time.
# The transcription worker, the health probe and the idle watcher all reach this
# state from different threadpool threads, so every load, unload and device
# switch happens under `model_lock`, and `model_in_use` keeps the watcher from
# unloading a model that is mid-transcription.
model: WhisperModel | None = None
model_lock = threading.RLock()
model_last_used: float = 0.0
model_in_use: int = 0
model_name: str = settings.whisper_model
model_device: str = settings.device
model_compute_type: str = settings.compute_type
model_cpu_fallback: bool = False
model_watcher_task: asyncio.Task[None] | None = None

T = TypeVar("T")

# CTranslate2 surfaces CUDA allocation failures as a plain RuntimeError, so the
# message is all there is to match on.
CUDA_OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudamalloc",
    "failed to allocate",
    "bad_alloc",
)


def _is_cuda_oom(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in CUDA_OOM_MARKERS)


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - treated as "no usable GPU"
        logger.debug("Could not query CUDA device count", exc_info=True)
        return 0


def _resolve_device() -> str:
    """Resolve DEVICE=auto to the device faster-whisper would pick itself."""

    if settings.device != "auto":
        return settings.device
    return "cuda" if _cuda_device_count() > 0 else "cpu"


def _cpu_compute_type() -> str:
    """Return a compute type CTranslate2 actually supports on this CPU.

    GPU compute types such as `int8_float16` have no CPU kernels, and a CPU run
    configured with one is silently downgraded, so pick a supported type here.
    """

    wanted = settings.cpu_fallback_compute_type
    try:
        import ctranslate2

        supported = set(ctranslate2.get_supported_compute_types("cpu"))
    except Exception:  # noqa: BLE001 - fall through to the configured value
        logger.debug("Could not query supported CPU compute types", exc_info=True)
        return wanted
    if wanted in supported:
        return wanted
    for candidate in ("int8", "int8_float32", "float32"):
        if candidate in supported:
            logger.warning(
                "CPU does not support compute_type=%s; using %s instead",
                wanted,
                candidate,
            )
            return candidate
    return wanted


def _cpu_fallback_target() -> tuple[str, str]:
    """Model and compute type to use when CUDA inference is not possible.

    WHISPER_MODEL is sized for the GPU; running it on CPU is several times
    slower than real time here, so the fallback uses a smaller multilingual
    model unless CPU_FALLBACK_MODEL is cleared.
    """

    return settings.cpu_fallback_model or settings.whisper_model, _cpu_compute_type()


def _build_model(name: str, device: str, compute_type: str) -> WhisperModel:
    logger.info(
        "Loading Whisper model %s device=%s compute_type=%s cpu_threads=%s",
        name,
        device,
        compute_type,
        settings.cpu_threads,
    )
    model_kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
    if settings.cpu_threads > 0:
        model_kwargs["cpu_threads"] = settings.cpu_threads
    return WhisperModel(name, **model_kwargs)


def _update_model_last_used_sync() -> None:
    global model_last_used
    model_last_used = time.monotonic()


def _restore_model_last_used(value: float) -> None:
    """Put the idle clock back, so a health probe cannot postpone the unload."""

    global model_last_used
    model_last_used = value


def _set_model(instance: WhisperModel, name: str, device: str, compute_type: str, *, fallback: bool) -> None:
    global model, model_name, model_device, model_compute_type, model_cpu_fallback
    model = instance
    model_name = name
    model_device = device
    model_compute_type = compute_type
    model_cpu_fallback = fallback
    _update_model_last_used_sync()


def _load_model_sync() -> WhisperModel:
    """Load the Whisper model synchronously (safe to call from a thread).

    Idempotent: will not reload if already loaded. The GPU is shared with other
    services, so a wake-up can land while another one holds the VRAM; when that
    happens the model is loaded for CPU inference instead of failing the request.
    """

    with model_lock:
        if model is not None:
            _update_model_last_used_sync()
            return model

        device = _resolve_device()
        name = settings.whisper_model
        compute_type = _cpu_compute_type() if device == "cpu" else settings.compute_type

        try:
            instance = _build_model(name, device, compute_type)
        except Exception as exc:
            if not (settings.cuda_oom_fallback_cpu and device != "cpu" and _is_cuda_oom(exc)):
                raise
            name, compute_type = _cpu_fallback_target()
            logger.warning(
                "CUDA is out of memory; falling back to CPU inference with model=%s compute_type=%s (%s)",
                name,
                compute_type,
                exc,
            )
            instance = _build_model(name, "cpu", compute_type)
            _set_model(instance, name, "cpu", compute_type, fallback=True)
            return instance

        _set_model(instance, name, device, compute_type, fallback=False)
        return instance


def _reload_on_cpu_sync() -> WhisperModel:
    """Drop the current model and load the CPU fallback in its place."""

    with model_lock:
        _unload_model_sync(force=True)
        name, compute_type = _cpu_fallback_target()
        instance = _build_model(name, "cpu", compute_type)
        _set_model(instance, name, "cpu", compute_type, fallback=True)
        return instance


@contextlib.contextmanager
def _model_in_use_guard():
    """Keep the idle watcher from unloading a model that is being used."""

    global model_in_use
    with model_lock:
        model_in_use += 1
    try:
        yield
    finally:
        with model_lock:
            model_in_use -= 1
            _update_model_last_used_sync()


def _run_with_cuda_fallback(run: Callable[[WhisperModel], T], *, what: str) -> T:
    """Run `run` against the loaded model, retrying on CPU after a CUDA OOM.

    CTranslate2 allocates lazily, so the GPU can still run out of memory well
    after the model itself loaded.
    """

    with _model_in_use_guard():
        instance = _load_model_sync()
        try:
            return run(instance)
        except Exception as exc:
            if not (settings.cuda_oom_fallback_cpu and model_device != "cpu" and _is_cuda_oom(exc)):
                raise
            logger.warning("CUDA is out of memory during %s; retrying on CPU (%s)", what, exc)
            instance = _reload_on_cpu_sync()
            return run(instance)


def _unload_model_sync(*, force: bool = False) -> bool:
    """Unload the Whisper model and free caches (best-effort)."""

    global model
    with model_lock:
        if model is None:
            return False
        if model_in_use > 0 and not force:
            return False
        logger.info("Unloading Whisper model %s (device=%s) from memory", model_name, model_device)
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
    return True


async def _model_idle_watcher() -> None:
    """Background task: unload model after configured idle timeout."""
    poll_interval = 5.0
    while True:
        try:
            if model is not None and settings.model_unload_seconds and settings.model_unload_seconds > 0:
                idle = time.monotonic() - model_last_used
                if idle >= settings.model_unload_seconds:
                    if await run_in_threadpool(_unload_model_sync):
                        logger.info(
                            "Whisper model unloaded after %.1fs idle (limit %.1fs)",
                            idle,
                            settings.model_unload_seconds,
                        )
        except Exception:
            logger.exception("Model idle watcher encountered an error")
        await asyncio.sleep(poll_interval)


def _model_was_loaded() -> bool:
    """True once a model has been loaded, whether or not it is resident now."""

    return model_last_used > 0.0


def _runtime_device() -> str:
    """Return the device actually chosen by faster-whisper.

    While the model sleeps this reports the device of the last load rather than
    the configured one, so a CPU fallback stays visible between requests.
    """

    if model is None:
        return model_device if _model_was_loaded() else settings.device
    impl = getattr(model, "model", None)
    actual = getattr(impl, "device", None) or getattr(model, "device", None)
    return str(actual) if actual else model_device


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
async def health(wake: Annotated[BoolArgument, Query()] = False) -> dict[str, float | str]:
    """Report service health.

    The probe transcribes a silent clip, which needs the model in memory. A
    sleeping model is left asleep unless `wake=true` or HEALTH_WAKE_MODEL is
    set, so that a health check running more often than MODEL_UNLOAD_SECONDS
    does not keep the model resident forever.
    """

    now = time.time()
    was_loaded = model is not None
    probe: dict[str, float | str | int]
    if was_loaded or wake or settings.health_wake_model:
        idle_before = model_last_used
        try:
            probe = await run_in_threadpool(_run_probe)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - we want the message in health output
            logger.exception("Health probe failed")
            probe = {"probe_error": str(exc)}
            status = "error"
        if was_loaded:
            # A probe of an already-loaded model must not postpone its unload.
            _restore_model_last_used(idle_before)
    else:
        probe = {}
        status = "ok"

    uptime_seconds = now - app_started_at
    idle_seconds = time.monotonic() - model_last_used if model_last_used else 0.0
    return {
        "status": status,
        "model": model_name if _model_was_loaded() else settings.whisper_model,
        "device": _runtime_device(),
        "compute_type": model_compute_type if _model_was_loaded() else settings.compute_type,
        "model_state": "loaded" if model is not None else "sleeping",
        "cpu_fallback": str(model_cpu_fallback),
        "model_unload_seconds": str(settings.model_unload_seconds),
        "model_idle_seconds": f"{idle_seconds:.1f}",
        "vad_filter": str(settings.vad_filter),
        "cpu_threads": str(settings.cpu_threads),
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
    task: Annotated[TaskArgument, Query()] = "transcribe",
    language: Annotated[LanguageArgument, Query()] = None,
    word_timestamps: Annotated[BoolArgument, Query()] = False,
    timeout_seconds: Annotated[TimeoutArgument, Query()] = None,
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
    language: Annotated[LanguageArgument, Form()] = None,
    prompt: str | None = Form(default=None),
    response_format: Annotated[ResponseFormatArgument, Form()] = "json",
    temperature: Annotated[TemperatureArgument, Form()] = 0.0,
    timeout_seconds: Annotated[TimeoutArgument, Form()] = None,
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
    response_format: Annotated[ResponseFormatArgument, Form()] = "json",
    temperature: Annotated[TemperatureArgument, Form()] = 0.0,
    timeout_seconds: Annotated[TimeoutArgument, Form()] = None,
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
