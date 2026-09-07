"""Typed Phase 3 configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from unsway.config import DevicePreference, DTypeName, ModelConfig

StorageDType = Literal["float16", "float32"]


@dataclass(frozen=True)
class Phase3Inputs:
    """Immutable behavioral inputs to feature discovery."""

    dataset_path: Path
    dataset_sha256: str
    predictions_path: Path
    predictions_sha256: str


@dataclass(frozen=True)
class ExtractionConfig:
    """Activation extraction settings."""

    layer: int
    location: str
    max_tokens_per_prompt: int
    shard_size_tokens: int
    storage_dtype: StorageDType
    splits: tuple[str, ...]
    max_examples_per_split: int | None

    @property
    def hook_name(self) -> str:
        """Return the canonical TransformerBridge hook name."""
        return f"blocks.{self.layer}.{self.location}"


@dataclass(frozen=True)
class SAEConfig:
    """Top-K sparse autoencoder hyperparameters."""

    expansion_factor: int
    k: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class FeatureAnalysisConfig:
    """Feature-ranking thresholds."""

    min_active_examples: int
    top_n_features: int


@dataclass(frozen=True)
class Phase3Output:
    """Phase 3 artifact and report destinations."""

    artifact_dir: Path
    extraction_report_path: Path
    training_report_path: Path
    feature_report_path: Path


@dataclass(frozen=True)
class Phase3Config:
    """Complete Phase 3 pipeline configuration."""

    seed: int
    inputs: Phase3Inputs
    model: ModelConfig
    extraction: ExtractionConfig
    sae: SAEConfig
    analysis: FeatureAnalysisConfig
    output: Phase3Output


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


def load_phase3_config(path: str | Path) -> Phase3Config:
    """Load and validate a Phase 3 YAML configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    inputs = _mapping(root.get("inputs"), "inputs")
    model = _mapping(root.get("model"), "model")
    extraction = _mapping(root.get("extraction"), "extraction")
    sae = _mapping(root.get("sae"), "sae")
    analysis = _mapping(root.get("analysis"), "analysis")
    output = _mapping(root.get("output"), "output")
    try:
        device = str(model.get("device", "auto"))
        dtype = str(model.get("dtype", "float32"))
        storage_dtype = str(extraction["storage_dtype"])
        if device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {dtype}")
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
        maximum = extraction.get("max_examples_per_split")
        config = Phase3Config(
            seed=int(experiment["seed"]),
            inputs=Phase3Inputs(
                dataset_path=Path(str(inputs["dataset_path"])),
                dataset_sha256=_checksum(inputs["dataset_sha256"], "Dataset"),
                predictions_path=Path(str(inputs["predictions_path"])),
                predictions_sha256=_checksum(inputs["predictions_sha256"], "Predictions"),
            ),
            model=ModelConfig(
                name=str(model["name"]),
                device=cast(DevicePreference, device),
                dtype=cast(DTypeName, dtype),
            ),
            extraction=ExtractionConfig(
                layer=int(extraction["layer"]),
                location=str(extraction["location"]),
                max_tokens_per_prompt=int(extraction["max_tokens_per_prompt"]),
                shard_size_tokens=int(extraction["shard_size_tokens"]),
                storage_dtype=cast(StorageDType, storage_dtype),
                splits=tuple(str(item) for item in extraction["splits"]),
                max_examples_per_split=None if maximum is None else int(maximum),
            ),
            sae=SAEConfig(
                expansion_factor=int(sae["expansion_factor"]),
                k=int(sae["k"]),
                epochs=int(sae["epochs"]),
                batch_size=int(sae["batch_size"]),
                learning_rate=float(sae["learning_rate"]),
                weight_decay=float(sae["weight_decay"]),
            ),
            analysis=FeatureAnalysisConfig(
                min_active_examples=int(analysis["min_active_examples"]),
                top_n_features=int(analysis["top_n_features"]),
            ),
            output=Phase3Output(
                artifact_dir=Path(str(output["artifact_dir"])),
                extraction_report_path=Path(str(output["extraction_report_path"])),
                training_report_path=Path(str(output["training_report_path"])),
                feature_report_path=Path(str(output["feature_report_path"])),
            ),
        )
    except KeyError as error:
        raise ValueError(f"Missing required Phase 3 field: {error.args[0]}") from error

    positive_integers = {
        "max_tokens_per_prompt": config.extraction.max_tokens_per_prompt,
        "shard_size_tokens": config.extraction.shard_size_tokens,
        "expansion_factor": config.sae.expansion_factor,
        "k": config.sae.k,
        "epochs": config.sae.epochs,
        "batch_size": config.sae.batch_size,
        "min_active_examples": config.analysis.min_active_examples,
        "top_n_features": config.analysis.top_n_features,
    }
    invalid = [name for name, value in positive_integers.items() if value <= 0]
    if invalid:
        raise ValueError(f"Phase 3 values must be positive: {', '.join(invalid)}")
    if config.extraction.layer < 0:
        raise ValueError("Extraction layer must be non-negative")
    if not config.extraction.splits:
        raise ValueError("At least one extraction split is required")
    if (
        config.extraction.max_examples_per_split is not None
        and config.extraction.max_examples_per_split <= 0
    ):
        raise ValueError("Maximum examples per split must be positive")
    if config.sae.learning_rate <= 0 or config.sae.weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    return config
