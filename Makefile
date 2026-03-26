UV ?= uv
PORT ?= 3373
APP = whisper_server.server:app

.PHONY: setup run dev lint format test docker-build docker-run compose

setup:
	$(UV) sync --python 3.13

run:
	$(UV) run uvicorn $(APP) --host 0.0.0.0 --port $(PORT)

dev:
	$(UV) run uvicorn $(APP) --host 0.0.0.0 --port $(PORT) --reload

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

test:
	$(UV) run pytest

docker-build:
	docker build -t whisper-server .

docker-run:
	docker run --rm --env-file .env -p $(PORT):$(PORT) whisper-server

compose:
	docker compose up --build
