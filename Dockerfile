FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    PORT=3373

COPY sample.env ./sample.env

EXPOSE 3373

CMD ["uv", "run", "uvicorn", "whisper_server.server:app", "--host", "0.0.0.0", "--port", "3373"]
