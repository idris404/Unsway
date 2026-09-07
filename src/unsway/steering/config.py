"""Typed configuration for leakage-safe Phase 4 steering experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from unsway.config import DevicePreference, DTypeName, ModelConfig

InterventionKind = Literal["sae_feature", "raw_neuron", "random_direction"]


@dataclass(frozen=True)
class Phase4Inputs:
    """Versioned behavioral and feature-discovery inputs."""

    dataset_path: Path
    dataset_sha256: str
    predictions_path: Path
    behavior_predictions_sha256: str
    sae_path: Path
    sae_sha256: str
    training_report_path: Path
    training_report_sha256: str
    feature_report_path: Path
    feature_report_sha256: str


@dataclass(frozen=True)
class InterventionConfig:
    """One causal direction to evaluate."""

    name: str
    kind: InterventionKind
    index: int | None
    seed: int | None
    match_strength_to: str | None


@dataclass(frozen=True)
class SteeringConfig:
    """Hook location, dose grid, and validation selection constraint."""

    layer: int
    location: str
    strengths: tuple[float, ...]
    max_initial_accuracy_drop: float
    max_batch_size: int
    max_batch_tokens: int
    interventions: tuple[InterventionConfig, ...]
    max_examples_per_split: int | None

    @property
    def hook_name(self) -> str:
        """Return the canonical TransformerBridge hook name."""
        return f"blocks.{self.layer}.{self.location}"


@dataclass(frozen=True)
class Phase4Output:
    """Prediction and report destinations."""

    prediction_dir: Path
    validation_report_path: Path
    test_report_path: Path


@dataclass(frozen=True)
class Phase4Config:
    """Complete Phase 4 experiment configuration."""

    seed: int
    inputs: Phase4Inputs
    model: ModelConfig
    steering: SteeringConfig
    output: Phase4Output


def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return cast(dict[str, Any], value)


def _checksum(value: object, name: str) -> str:
    checksum = str(value)
    if len(checksum) != 64:
        raise ValueError(f"{name} SHA-256 must contain 64 characters")
    try:
        int(checksum, 16)
    except ValueError as error:
        raise ValueError(f"{name} SHA-256 must be hexadecimal") from error
    return checksum


def _intervention(value: object) -> InterventionConfig:
    raw = _mapping(value, "intervention")
    kind = str(raw["kind"])
    if kind not in {"sae_feature", "raw_neuron", "random_direction"}:
        raise ValueError(f"Unsupported intervention kind: {kind}")
    index = raw.get("index")
    seed = raw.get("seed")
    if kind == "random_direction" and seed is None:
        raise ValueError("Random directions require a seed")
    if kind != "random_direction" and index is None:
        raise ValueError(f"{kind} interventions require an index")
    return InterventionConfig(
        name=str(raw["name"]),
        kind=cast(InterventionKind, kind),
        index=None if index is None else int(index),
        seed=None if seed is None else int(seed),
        match_strength_to=(
            None if raw.get("match_strength_to") is None else str(raw["match_strength_to"])
        ),
    )


def load_phase4_config(path: str | Path) -> Phase4Config:
    """Load and validate a Phase 4 YAML configuration."""
    root = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    inputs = _mapping(root.get("inputs"), "inputs")
    model = _mapping(root.get("model"), "model")
    steering = _mapping(root.get("steering"), "steering")
    output = _mapping(root.get("output"), "output")
    device = str(model.get("device", "auto"))
    dtype = str(model.get("dtype", "float32"))
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {device}")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype: {dtype}")
    maximum = steering.get("max_examples_per_split")
    config = Phase4Config(
        seed=int(experiment["seed"]),
        inputs=Phase4Inputs(
            dataset_path=Path(str(inputs["dataset_path"])),
            dataset_sha256=_checksum(inputs["dataset_sha256"], "Dataset"),
            predictions_path=Path(str(inputs["predictions_path"])),
            behavior_predictions_sha256=_checksum(
                inputs["behavior_predictions_sha256"], "Behavior predictions"
            ),
            sae_path=Path(str(inputs["sae_path"])),
            sae_sha256=_checksum(inputs["sae_sha256"], "SAE"),
            training_report_path=Path(str(inputs["training_report_path"])),
            training_report_sha256=_checksum(inputs["training_report_sha256"], "Training report"),
            feature_report_path=Path(str(inputs["feature_report_path"])),
            feature_report_sha256=_checksum(inputs["feature_report_sha256"], "Feature report"),
        ),
        model=ModelConfig(
            name=str(model["name"]),
            device=cast(DevicePreference, device),
            dtype=cast(DTypeName, dtype),
        ),
        steering=SteeringConfig(
            layer=int(steering["layer"]),
            location=str(steering["location"]),
            strengths=tuple(float(value) for value in steering["strengths"]),
            max_initial_accuracy_drop=float(steering["max_initial_accuracy_drop"]),
            max_batch_size=int(steering["max_batch_size"]),
            max_batch_tokens=int(steering["max_batch_tokens"]),
            interventions=tuple(_intervention(value) for value in steering["interventions"]),
            max_examples_per_split=None if maximum is None else int(maximum),
        ),
        output=Phase4Output(
            prediction_dir=Path(str(output["prediction_dir"])),
            validation_report_path=Path(str(output["validation_report_path"])),
            test_report_path=Path(str(output["test_report_path"])),
        ),
    )
    if config.steering.layer < 0:
        raise ValueError("Steering layer must be non-negative")
    if not config.steering.interventions:
        raise ValueError("At least one intervention is required")
    names = [item.name for item in config.steering.interventions]
    if len(names) != len(set(names)):
        raise ValueError("Intervention names must be unique")
    for intervention in config.steering.interventions:
        if intervention.match_strength_to is not None:
            if intervention.match_strength_to not in names:
                raise ValueError("Matched intervention references an unknown name")
            if intervention.match_strength_to == intervention.name:
                raise ValueError("An intervention cannot match its own strength")
    if not config.steering.strengths or 0.0 not in config.steering.strengths:
        raise ValueError("Steering strengths must include zero")
    if not any(value > 0 for value in config.steering.strengths):
        raise ValueError("Steering strengths must include a positive dose")
    if any(not math.isfinite(value) for value in config.steering.strengths):
        raise ValueError("Steering strengths must be finite")
    if len(config.steering.strengths) != len(set(config.steering.strengths)):
        raise ValueError("Steering strengths must be unique")
    if config.steering.max_batch_size <= 0 or config.steering.max_batch_tokens <= 0:
        raise ValueError("Batch limits must be positive")
    if not 0 <= config.steering.max_initial_accuracy_drop <= 1:
        raise ValueError("Maximum initial accuracy drop must be between zero and one")
    if (
        config.steering.max_examples_per_split is not None
        and config.steering.max_examples_per_split <= 0
    ):
        raise ValueError("Maximum examples per split must be positive")
    for intervention in config.steering.interventions:
        if intervention.index is not None and intervention.index < 0:
            raise ValueError("Intervention indices must be non-negative")
        match = intervention.match_strength_to
        if match is not None:
            target = next(item for item in config.steering.interventions if item.name == match)
            if target.match_strength_to is not None:
                raise ValueError("Matched strengths cannot form a chain")
    return config
