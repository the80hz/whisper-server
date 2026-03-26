# Whisper Server

FastAPI-based microservice that wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for lightweight, production-friendly speech-to-text workloads. The project uses [uv](https://docs.astral.sh/uv/) for dependency management and targets Python 3.13.

## Features

- Single `/transcribe` endpoint accepting audio uploads via `multipart/form-data`.
- One-time in-memory loading of the configured Whisper model for low-latency responses.
- Built-in FIFO queue with a single worker to avoid concurrent model conflicts.
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
| `MAX_UPLOAD_MB` | `50` | Default upload size limit for `/transcribe`. |

## `/transcribe` Arguments

The endpoint supports query parameters in addition to file upload:

- `task`: `transcribe` (default) or `translate`
- `language`: language code hint (for example `ru`, `en`)
- `word_timestamps`: `true/false` to include per-word timestamps
- `timeout_seconds`: override request timeout for a single call
- `max_file_size_mb`: override max file size for a single call

## Docker & Compose

Build and run with Docker:

```bash
docker build -t whisper-server .
docker run --env-file .env -p ${PORT:-3373}:${PORT:-3373} whisper-server
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
