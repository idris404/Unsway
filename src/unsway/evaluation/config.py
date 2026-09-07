"""Typed configuration for the Phase 2 behavioral baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from unsway.config import DevicePreference, DTypeName, ModelConfig


@dataclass(frozen=True)
class DatasetConfig:
    """Input dataset path and expected content identity."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class EvaluationConfig:
    """Memory-bounded inference settings."""

    max_batch_size: int
    max_batch_tokens: int


@dataclass(frozen=True)
class BaselineOutputConfig:
    """Prediction and aggregate-report destinations."""

    predictions_path: Path
    report_path: Path


@dataclass(frozen=True)
class Phase2Config:
    """Complete Phase 2 baseline configuration."""

    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: BaselineOutputConfig


def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return cast(dict[str, Any], value)


def load_phase2_config(path: str | Path) -> Phase2Config:
    """Load and validate the Phase 2 YAML configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    dataset = _mapping(root.get("dataset"), "dataset")
    model = _mapping(root.get("model"), "model")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    output = _mapping(root.get("output"), "output")
    try:
        device = str(model.get("device", "auto"))
        dtype = str(model.get("dtype", "float32"))
        if device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {dtype}")
        config = Phase2Config(
            seed=int(experiment["seed"]),
            dataset=DatasetConfig(path=Path(str(dataset["path"])), sha256=str(dataset["sha256"])),
            model=ModelConfig(
                name=str(model["name"]),
                device=cast(DevicePreference, device),
                dtype=cast(DTypeName, dtype),
            ),
            evaluation=EvaluationConfig(
                max_batch_size=int(evaluation["max_batch_size"]),
                max_batch_tokens=int(evaluation["max_batch_tokens"]),
            ),
            output=BaselineOutputConfig(
                predictions_path=Path(str(output["predictions_path"])),
                report_path=Path(str(output["report_path"])),
            ),
        )
    except KeyError as error:
        raise ValueError(f"Missing required Phase 2 field: {error.args[0]}") from error

    if len(config.dataset.sha256) != 64:
        raise ValueError("Dataset SHA-256 must contain 64 characters")
    try:
        int(config.dataset.sha256, 16)
    except ValueError as error:
        raise ValueError("Dataset SHA-256 must be hexadecimal") from error
    if config.evaluation.max_batch_size <= 0:
        raise ValueError("Maximum batch size must be positive")
    if config.evaluation.max_batch_tokens <= 0:
        raise ValueError("Maximum batch tokens must be positive")
    return config
