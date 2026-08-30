.PHONY: install test test-all lint typecheck check phase0

install:
	uv sync --extra dev

test:
	uv run pytest

test-all:
	uv run pytest -m "integration or not integration"

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

check: lint typecheck test

phase0:
	uv run unsway-phase0 --config configs/phase0.yaml

