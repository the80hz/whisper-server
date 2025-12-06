FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS build

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml README.md sample.env ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PORT=3373 \
    PYTHONPATH="/app/src" \
    LD_LIBRARY_PATH="/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

EXPOSE 3373

CMD ["uvicorn", "whisper_server.server:app", "--host", "0.0.0.0", "--port", "3373"]
