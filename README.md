# Whisper Server

FastAPI-based microservice that wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for lightweight, production-friendly speech-to-text workloads. The project uses [uv](https://docs.astral.sh/uv/) for dependency management and targets Python 3.13.

## Features

- Single `/transcribe` endpoint accepting audio uploads via `multipart/form-data`.
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints for drop-in local API usage.
- Lazy model loading with an idle sleep: the model is unloaded after
  `MODEL_UNLOAD_SECONDS` and reloaded on the next request, so a shared GPU is
  only held while there is work.
- Automatic CPU fallback when CUDA runs out of memory, using a CPU-sized model
  and a CPU-supported compute type.
- Built-in FIFO queue with a single worker to avoid concurrent model conflicts.
- Optional bearer-token authentication via `API_TOKEN`.
- Backward-compatible `API_KEY` alias for older bratishkabot whisper deployments.
- JSON, plain text, SRT, VTT, and verbose JSON responses.
- Configurable via environment variables (`sample.env` provided).
- Ready-to-ship Dockerfile plus `docker compose` definition and Makefile shortcuts.

## Quick Start

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) and ensure Python 3.13 is available.
2. Copy the example environment file: `cp sample.env .env` and tweak values as needed.
3. Install dependencies with `uv sync`.
4. Launch the API:

   ```bash
   uv run uvicorn whisper_server.server:app --host 0.0.0.0 --port ${PORT:-3373}
   ```

5. Transcribe audio via curl:

   ```bash
   curl -X POST "http://localhost:3373/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/audio.wav"
   ```

6. Or use the OpenAI-compatible endpoint:

   ```bash
   curl -X POST "http://localhost:3373/v1/audio/transcriptions" \
     -H "Authorization: Bearer ${API_TOKEN}" \
     -F "file=@/path/to/audio.wav" \
     -F "model=whisper-1" \
     -F "language=ru" \
     -F "response_format=verbose_json"
   ```

Alternatively, use the Makefile helpers: `make setup` and `make run`.

## Configuration

Environment variables (see `sample.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `3373` | Port exposed by uvicorn and Docker image. |
| `WHISPER_MODEL` | `large-v3-turbo` | Model name accepted by faster-whisper. |
| `LOG_LEVEL` | `INFO` | Root logging verbosity. |
| `LOG_FILE` | `logs/whisper.log` | Path for persistent application logs (directory created automatically). |
| `COMPUTE_TYPE` | `int8` | faster-whisper compute type (e.g., `int8`, `int8_float16`, `float16`). |
| `DEVICE` | `auto` | Device hint passed to faster-whisper (`auto`, `cpu`, `cuda`). |
| `QUEUE_MAX_SIZE` | `8` | Maximum number of pending transcription jobs in the queue. |
| `DEFAULT_TIMEOUT_SECONDS` | `180` | Per-request timeout when `timeout_seconds` is not provided. |
| `VAD_FILTER` | `true` | Enable faster-whisper VAD filtering before transcription. |
| `VAD_THRESHOLD` | `0.5` | Silero VAD speech probability threshold. |
| `VAD_MIN_SILENCE_DURATION_MS` | `500` | Silence duration used to split speech regions. |
| `VAD_SPEECH_PAD_MS` | `200` | Audio padding retained around detected speech. |
| `CONDITION_ON_PREVIOUS_TEXT` | `false` | Feed the previous window into the next one; disabled by default to prevent repetition loops. |
| `REPETITION_PENALTY` | `1.1` | Penalty applied to tokens that were already generated. |
| `NO_REPEAT_NGRAM_SIZE` | `3` | Prevent repeated n-grams of this size; `0` disables the restriction. |
| `COMPRESSION_RATIO_THRESHOLD` | `2.2` | Treat highly compressible (usually repetitive) output as a failed decoding attempt. |
| `LOG_PROB_THRESHOLD` | `-1.0` | Retry decoding when average token probability is too low. |
| `NO_SPEECH_THRESHOLD` | `0.6` | Probability threshold used to classify a window as silence. |
| `TEMPERATURE_FALLBACK` | `true` | Retry zero-temperature decoding at increasing temperatures when quality thresholds fail. |
| `HALLUCINATION_SILENCE_THRESHOLD` | `1.0` | Skip silence around suspected hallucinations; `0` disables this behavior. |
| `CPU_THREADS` | `0` | CPU worker threads passed to faster-whisper when greater than `0`; `0` means auto. |
| `MODEL_UNLOAD_SECONDS` | `60` | Idle seconds before the model is unloaded and its VRAM released. Set `0` to keep it loaded. |
| `HEALTH_WAKE_MODEL` | `false` | Let `/health` wake a sleeping model for its probe. See [Idle sleep and CPU fallback](#idle-sleep-and-cpu-fallback). |
| `CUDA_OOM_FALLBACK_CPU` | `true` | Fall back to CPU inference when CUDA is out of memory. |
| `CPU_FALLBACK_MODEL` | `small` | Model used by that fallback; empty keeps `WHISPER_MODEL`. |
| `CPU_FALLBACK_COMPUTE_TYPE` | `int8` | Compute type for CPU runs; GPU-only types are not supported on CPU. |
| `MAX_UPLOAD_MB` | `50` | Default upload size limit for `/transcribe`. |
| `API_TOKEN` | unset | Optional bearer token required for all transcription endpoints when set. |
| `API_KEY` | unset | Compatibility alias for `API_TOKEN`; `API_TOKEN` takes precedence. |

## Idle Sleep and CPU Fallback

The model is not kept in memory between requests. After `MODEL_UNLOAD_SECONDS`
without work it is unloaded, its VRAM is released, and the next request loads it
again. On a GPU shared with other services this is what lets them use the card
while no transcription is running; the cost is the model load time on the first
request after a sleep.

Because the GPU is shared, a wake-up can land while another process holds the
VRAM. When `CUDA_OOM_FALLBACK_CPU` is set, a CUDA out-of-memory error does not
fail the request: the model is loaded for CPU inference instead, and the
transcription continues. CTranslate2 allocates lazily, so this is handled both
when the model loads and when the GPU runs out of memory mid-transcription.

A CPU run is not the GPU run with a different device flag:

- **Model.** `WHISPER_MODEL` is sized for the GPU. On a 4-core container CPU
  (Xeon E5-2690 v4, `int8`, Russian speech) `large-v3-turbo` measured a real-time
  factor of 0.88 against 0.38 for `small`, which leaves almost no headroom before
  `DEFAULT_TIMEOUT_SECONDS` once decoding retries at higher temperatures. The
  fallback therefore loads `CPU_FALLBACK_MODEL`, a smaller multilingual model,
  unless that value is empty. `base` is faster still but drops digits, and the
  `distil-*` models are English-only.
- **Compute type.** GPU compute types such as `int8_float16` and `float16` have
  no CPU kernels in CTranslate2. Any CPU run uses `CPU_FALLBACK_COMPUTE_TYPE`,
  checked against `ctranslate2.get_supported_compute_types("cpu")` and downgraded
  to a supported type if needed. This also applies to an explicit `DEVICE=cpu`
  and to `DEVICE=auto` on a host without a GPU, where the configured
  `WHISPER_MODEL` is still honoured.

The fallback is per load: the next wake-up tries the GPU again.

`/health` reports the current state:

| Field | Meaning |
| --- | --- |
| `model_state` | `loaded` or `sleeping` |
| `device` / `compute_type` | What the loaded model actually uses, not what is configured |
| `cpu_fallback` | `True` when the model is on CPU after a CUDA out-of-memory error |
| `model_idle_seconds` / `model_unload_seconds` | How close the model is to being unloaded |

The probe transcribes a silent clip, which needs the model in memory. A sleeping
model is left asleep and `/health` still answers `"status": "ok"`, so a monitor
polling more often than `MODEL_UNLOAD_SECONDS` does not pin the model in VRAM.
`GET /health?wake=true` forces a full probe, and `HEALTH_WAKE_MODEL=true` makes
that the default. A probe of an already-loaded model does not postpone its
unload.

## `/transcribe` Arguments

The endpoint supports query parameters in addition to file upload:

- `task`: `transcribe` (default) or `translate`
- `language`: language code hint (for example `ru`, `en`)
- `word_timestamps`: `true/false` to include per-word timestamps
- `timeout_seconds`: override request timeout for a single call

When `API_TOKEN` or `API_KEY` is set, include `Authorization: Bearer <token>`.

This endpoint is compatible with the current `bratishkabot` remote STT client:

- `GET /health` returns a JSON object with `status`.
- `POST /transcribe?language=ru` accepts a Telegram `voice.ogg` upload in multipart field `file`.
- The response includes a top-level `text` field.

## OpenAI-Compatible API

`/v1/audio/transcriptions` accepts OpenAI-style multipart form fields:

- `file`: audio or video file upload
- `model`: accepted for compatibility; the server uses `WHISPER_MODEL`
- `language`: optional language code hint, for example `ru` or `en`
- `prompt`: optional initial prompt
- `response_format`: `json`, `text`, `srt`, `vtt`, or `verbose_json`
- `temperature`: decoding temperature, default `0`
- `timeout_seconds`: optional server-side timeout override

`/v1/audio/translations` has the same shape and runs Whisper's `translate` task.

Examples:

```bash
curl -X POST "http://whisper-gpu:3373/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -F "file=@meeting.m4a" \
  -F "model=whisper-1" \
  -F "language=ru" \
  -F "response_format=text"
```

```bash
curl -X POST "http://whisper-gpu:3373/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -F "file=@lecture.mp4" \
  -F "model=whisper-1" \
  -F "response_format=srt"
```

## Docker & Compose

Build and run with Docker:

```bash
docker build -t whisper-server .
docker run --env-file .env -p ${PORT:-3373}:${PORT:-3373} whisper-server
```

The default image is CPU-only to keep uploads smaller. To build an image with
CUDA runtime libraries bundled:

```bash
docker build --build-arg INSTALL_GPU=true -t whisper-server:gpu .
```

Or use Compose:

```bash
docker compose up --build
```

The compose service loads `.env`, maps the configured port, and can be extended with volumes for cached models if desired.

For NVIDIA GPU hosts (Linux), use the GPU override:

```bash
docker compose -f compose.yml -f compose.gpu.yml up --build
```

If you use the published image instead of building locally, pull the GPU tag:

```bash
docker compose -f compose.yml -f compose.gpu.yml pull
docker compose -f compose.yml -f compose.gpu.yml up
```

The GPU image must contain `nvidia-cudnn-cu12`. If CUDA crashes with a missing
`libcudnn_ops.so` error, rebuild or pull the `:gpu` image rather than reusing
`:latest`.

GitHub Actions builds both variants on pull requests. Pushes to `main` publish
multi-architecture images to the GitHub Container Registry as
`ghcr.io/the80hz/whisper-api:latest` for CPU and `ghcr.io/the80hz/whisper-api:gpu`
for CUDA. No registry secrets are needed: the workflow authenticates with the
built-in `GITHUB_TOKEN` and `packages: write`, and the package inherits this
repository's visibility, so the images pull anonymously.

For LAN, Tailscale, or OpenVPN usage, bind the service on the GPU host and call it by its private address, for example `http://gpu-box:3373` or `http://100.x.y.z:3373`. Set `API_TOKEN` when the port is reachable by other machines.

## Project Layout

```text
.
├── compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
├── sample.env
└── src/
    └── whisper_server/
        ├── __init__.py
        ├── config.py
        └── server.py
```

## Development Notes

- Linting: `uv run ruff check .`
- Tests: `uv run pytest`
- Use `uv lock` to generate a lockfile if you need a deterministic dependency snapshot.
