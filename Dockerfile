FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS build

WORKDIR /app

ARG INSTALL_GPU=false

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN if [ "$INSTALL_GPU" = "true" ]; then \
        uv sync --frozen --no-dev --extra gpu; \
    else \
        uv sync --frozen --no-dev; \
    fi

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml README.md sample.env ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PORT=3373 \
    PYTHONPATH="/app/src" \
    LD_LIBRARY_PATH="/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib"

EXPOSE 3373

CMD ["sh", "-c", "exec uvicorn whisper_server.server:app --host 0.0.0.0 --port ${PORT:-3373}"]
