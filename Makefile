.PHONY: install test test-all lint typecheck check phase0 phase1 phase2 phase3-smoke phase5 phase6-data phase6b phase6c-data phase6c-baseline phase6c-extract

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

phase1:
	uv run unsway-phase1 --config configs/phase1.yaml

phase2:
	uv run unsway-phase2 --config configs/phase2.yaml

phase3-smoke:
	uv run unsway-phase3 --config configs/phase3_smoke.yaml --stage all

phase5:
	uv run unsway-phase5 --config configs/phase5.yaml

phase6-data:
	uv run unsway-phase6 --config configs/phase6.yaml --stage data

phase6b:
	uv run unsway-phase6 --config configs/phase6.yaml --stage phase6b

phase6c-data:
	uv run unsway-phase6 --config configs/phase6c.yaml --stage data

phase6c-baseline:
	uv run unsway-phase6 --config configs/phase6c.yaml --stage baseline

phase6c-extract:
	uv run unsway-phase6 --config configs/phase6.yaml --stage extract --eligibility-config configs/phase6c.yaml
