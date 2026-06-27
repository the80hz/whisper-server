# Whisper Server

FastAPI-based microservice that wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for lightweight, production-friendly speech-to-text workloads. The project uses [uv](https://docs.astral.sh/uv/) for dependency management and targets Python 3.13.

## Features

- Single `/transcribe` endpoint accepting audio uploads via `multipart/form-data`.
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints for drop-in local API usage.
- One-time in-memory loading of the configured Whisper model for low-latency responses.
- Built-in FIFO queue with a single worker to avoid concurrent model conflicts.
- Optional bearer-token authentication via `API_TOKEN`.
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
| `MODEL_UNLOAD_SECONDS` | `600` | Idle seconds before unloading the model. Set `0` to keep it loaded. |
| `MAX_UPLOAD_MB` | `50` | Default upload size limit for `/transcribe`. |
| `API_TOKEN` | unset | Optional bearer token required for all transcription endpoints when set. |

## `/transcribe` Arguments

The endpoint supports query parameters in addition to file upload:

- `task`: `transcribe` (default) or `translate`
- `language`: language code hint (for example `ru`, `en`)
- `word_timestamps`: `true/false` to include per-word timestamps
- `timeout_seconds`: override request timeout for a single call

When `API_TOKEN` is set, include `Authorization: Bearer <token>`.

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

GitHub Actions builds both variants on pull requests. Pushes to `main` publish
multi-architecture images to Docker Hub as `the80hz/whisper-api:latest` for CPU
and `the80hz/whisper-api:gpu` for CUDA. Configure repository secrets
`DOCKER_USERNAME` and `DOCKER_TOKEN` before publishing.

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
