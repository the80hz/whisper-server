UV ?= uv
PORT ?= 3373

.PHONY: setup setup-gpu setup-gigaam setup-gigaam-cpu run dev check format test

# One variant at a time; each `setup-*` replaces the previous one.
setup:
	$(UV) sync

setup-gpu:
	$(UV) sync --extra gpu

setup-gigaam:
	$(UV) sync --extra gigaam

setup-gigaam-cpu:
	$(UV) sync --extra gigaam-cpu

run:
	PORT=$(PORT) $(UV) run --no-sync whisper-server

dev:
	$(UV) run --no-sync uvicorn whisper_server.server:app --host 0.0.0.0 --port $(PORT) --reload

check:
	$(UV) run --no-sync ruff format --check . && $(UV) run --no-sync ruff check . && $(UV) run --no-sync pytest

format:
	$(UV) run --no-sync ruff format .

test:
	$(UV) run --no-sync pytest
