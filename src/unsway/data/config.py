"""Typed configuration for Phase 1 dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class SourceConfig:
    """Remote source identity and integrity information."""

    name: str
    url: str
    sha256: str
    path: Path


@dataclass(frozen=True)
class OutputConfig:
    """Generated dataset and manifest paths."""

    dataset_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer and context-budget settings."""

    name: str
    max_prompt_tokens: int


@dataclass(frozen=True)
class BuildConfig:
    """Filtering, splitting, and reproducibility settings."""

    seed: int
    allowed_datasets: tuple[str, ...]
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class Phase1Config:
    """Complete Phase 1 configuration."""

    source: SourceConfig
    output: OutputConfig
    tokenizer: TokenizerConfig
    build: BuildConfig


def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return cast(dict[str, Any], value)


def load_phase1_config(path: str | Path) -> Phase1Config:
    """Load and validate the Phase 1 YAML configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    source = _mapping(root.get("source"), "source")
    output = _mapping(root.get("output"), "output")
    tokenizer = _mapping(root.get("tokenizer"), "tokenizer")
    build = _mapping(root.get("build"), "build")

    try:
        config = Phase1Config(
            source=SourceConfig(
                name=str(source["name"]),
                url=str(source["url"]),
                sha256=str(source["sha256"]),
                path=Path(str(source["path"])),
            ),
            output=OutputConfig(
                dataset_path=Path(str(output["dataset_path"])),
                manifest_path=Path(str(output["manifest_path"])),
            ),
            tokenizer=TokenizerConfig(
                name=str(tokenizer["name"]),
                max_prompt_tokens=int(tokenizer["max_prompt_tokens"]),
            ),
            build=BuildConfig(
                seed=int(build["seed"]),
                allowed_datasets=tuple(str(item) for item in build["allowed_datasets"]),
                train_fraction=float(build["train_fraction"]),
                validation_fraction=float(build["validation_fraction"]),
                test_fraction=float(build["test_fraction"]),
            ),
        )
    except KeyError as error:
        raise ValueError(f"Missing required Phase 1 field: {error.args[0]}") from error

    if len(config.source.sha256) != 64:
        raise ValueError("Source SHA-256 must contain 64 hexadecimal characters")
    try:
        int(config.source.sha256, 16)
    except ValueError as error:
        raise ValueError("Source SHA-256 must be hexadecimal") from error
    if config.tokenizer.max_prompt_tokens <= 0:
        raise ValueError("Maximum prompt token count must be positive")
    if not config.build.allowed_datasets:
        raise ValueError("At least one source dataset must be allowed")
    split_total = (
        config.build.train_fraction + config.build.validation_fraction + config.build.test_fraction
    )
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {split_total}")
    if (
        min(
            config.build.train_fraction,
            config.build.validation_fraction,
            config.build.test_fraction,
        )
        <= 0
    ):
        raise ValueError("Every split fraction must be positive")
    return config
