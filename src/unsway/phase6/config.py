"""Typed, pre-registered configuration for Phase 6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml


class ArtifactFormat(StrEnum):
    """Supported external question artifact formats."""

    JSONL = "jsonl"
    ZIP_JSONL = "zip_jsonl"
    PARQUET = "parquet"


@dataclass(frozen=True)
class ArtifactConfig:
    """One checksum-pinned external data artifact."""

    source: str
    url: str
    sha256: str
    path: Path
    format: ArtifactFormat
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetConfig:
    """Fresh-dataset construction and partitioning rules."""

    seed: int
    max_examples_per_source: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    tokenizer_name: str
    max_prompt_tokens: int
    phase1_dataset_path: Path
    dataset_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ProtocolConfig:
    """Scientific choices that must be frozen before the new test is opened."""

    experiment_id: str
    model_name: str
    hook_points: tuple[str, ...]
    direction_families: tuple[str, ...]
    strengths: tuple[float, ...]
    diagnostic_negative_strengths: tuple[float, ...]
    train_cv_folds: int
    max_caa_layers_for_validation: int
    min_test_eligible: int
    min_effect_pp: float
    max_accuracy_drop_pp: float
    min_sources_consistent: int
    bootstrap_replicates: int
    primary_metric: str
    refuse_test_overwrite: bool
    manifest_path: Path


@dataclass(frozen=True)
class Phase6Config:
    """Complete Phase 6 configuration."""

    artifacts: tuple[ArtifactConfig, ...]
    dataset: DatasetConfig
    protocol: ProtocolConfig


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    return value


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error


def load_phase6_config(path: str | Path) -> Phase6Config:
    """Load and validate the pre-registered Phase 6 configuration."""
    raw = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "Phase 6 config")
    artifact_rows = _sequence(raw.get("artifacts"), "artifacts")
    artifacts: list[ArtifactConfig] = []
    for index, raw_artifact in enumerate(artifact_rows):
        artifact = _mapping(raw_artifact, f"artifacts[{index}]")
        try:
            parsed = ArtifactConfig(
                source=str(artifact["source"]),
                url=str(artifact["url"]),
                sha256=str(artifact["sha256"]),
                path=Path(str(artifact["path"])),
                format=ArtifactFormat(str(artifact["format"])),
                members=tuple(str(member) for member in artifact.get("members", [])),
            )
        except KeyError as error:
            raise ValueError(f"Missing artifact field: {error.args[0]}") from error
        _validate_sha256(parsed.sha256, f"artifacts[{index}].sha256")
        if parsed.format is ArtifactFormat.ZIP_JSONL and not parsed.members:
            raise ValueError("zip_jsonl artifacts require at least one member")
        if parsed.format is not ArtifactFormat.ZIP_JSONL and parsed.members:
            raise ValueError("Only zip_jsonl artifacts may declare members")
        artifacts.append(parsed)

    dataset = _mapping(raw.get("dataset"), "dataset")
    protocol = _mapping(raw.get("protocol"), "protocol")
    try:
        parsed_dataset = DatasetConfig(
            seed=int(dataset["seed"]),
            max_examples_per_source=int(dataset["max_examples_per_source"]),
            train_fraction=float(dataset["train_fraction"]),
            validation_fraction=float(dataset["validation_fraction"]),
            test_fraction=float(dataset["test_fraction"]),
            tokenizer_name=str(dataset["tokenizer_name"]),
            max_prompt_tokens=int(dataset["max_prompt_tokens"]),
            phase1_dataset_path=Path(str(dataset["phase1_dataset_path"])),
            dataset_path=Path(str(dataset["dataset_path"])),
            manifest_path=Path(str(dataset["manifest_path"])),
        )
        parsed_protocol = ProtocolConfig(
            experiment_id=str(protocol["experiment_id"]),
            model_name=str(protocol["model_name"]),
            hook_points=tuple(str(item) for item in protocol["hook_points"]),
            direction_families=tuple(str(item) for item in protocol["direction_families"]),
            strengths=tuple(float(item) for item in protocol["strengths"]),
            diagnostic_negative_strengths=tuple(
                float(item) for item in protocol["diagnostic_negative_strengths"]
            ),
            train_cv_folds=int(protocol["train_cv_folds"]),
            max_caa_layers_for_validation=int(protocol["max_caa_layers_for_validation"]),
            min_test_eligible=int(protocol["min_test_eligible"]),
            min_effect_pp=float(protocol["min_effect_pp"]),
            max_accuracy_drop_pp=float(protocol["max_accuracy_drop_pp"]),
            min_sources_consistent=int(protocol["min_sources_consistent"]),
            bootstrap_replicates=int(protocol["bootstrap_replicates"]),
            primary_metric=str(protocol["primary_metric"]),
            refuse_test_overwrite=bool(protocol["refuse_test_overwrite"]),
            manifest_path=Path(str(protocol["manifest_path"])),
        )
    except KeyError as error:
        raise ValueError(f"Missing Phase 6 field: {error.args[0]}") from error

    fractions = (
        parsed_dataset.train_fraction,
        parsed_dataset.validation_fraction,
        parsed_dataset.test_fraction,
    )
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError("Phase 6 split fractions must be positive and sum to 1")
    if parsed_dataset.max_examples_per_source <= 0 or parsed_dataset.max_prompt_tokens <= 0:
        raise ValueError("Dataset size and prompt-token limits must be positive")
    if len({artifact.source for artifact in artifacts}) < 2:
        raise ValueError("Phase 6 requires at least two independent source families")
    if len(parsed_protocol.hook_points) != 12:
        raise ValueError("Phase 6 pre-registers all 12 GPT-2 small residual layers")
    if parsed_protocol.strengths[0] != 0 or any(
        right <= left
        for left, right in zip(
            parsed_protocol.strengths, parsed_protocol.strengths[1:], strict=False
        )
    ):
        raise ValueError("Steering strengths must start at zero and increase strictly")
    if not parsed_protocol.diagnostic_negative_strengths or any(
        strength >= 0 for strength in parsed_protocol.diagnostic_negative_strengths
    ):
        raise ValueError("Diagnostic reverse-direction strengths must all be negative")
    if parsed_protocol.train_cv_folds < 2:
        raise ValueError("At least two train cross-validation folds are required")
    if parsed_protocol.min_test_eligible <= 0 or parsed_protocol.bootstrap_replicates < 1_000:
        raise ValueError("Test eligibility and bootstrap counts are too small")
    if parsed_protocol.min_effect_pp <= 0 or parsed_protocol.max_accuracy_drop_pp < 0:
        raise ValueError("Effect and accuracy guardrails must be non-negative")
    if (
        not 1
        <= parsed_protocol.min_sources_consistent
        <= len({artifact.source for artifact in artifacts})
    ):
        raise ValueError("Source-consistency requirement is incompatible with the artifacts")
    return Phase6Config(tuple(artifacts), parsed_dataset, parsed_protocol)


def protocol_payload(config: Phase6Config) -> dict[str, Any]:
    """Return the path-independent scientific protocol used for hashing."""
    return {
        "schema_version": 1,
        "experiment_id": config.protocol.experiment_id,
        "artifacts": [
            {
                "source": artifact.source,
                "url": artifact.url,
                "sha256": artifact.sha256,
                "format": artifact.format.value,
                "members": list(artifact.members),
            }
            for artifact in config.artifacts
        ],
        "dataset": {
            key: value for key, value in asdict(config.dataset).items() if not key.endswith("_path")
        },
        "protocol": {
            key: value for key, value in asdict(config.protocol).items() if key != "manifest_path"
        },
    }


def protocol_sha256(config: Phase6Config) -> str:
    """Hash every scientific choice that must remain fixed through testing."""
    encoded = json.dumps(protocol_payload(config), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
