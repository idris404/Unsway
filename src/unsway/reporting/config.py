"""Typed configuration for Phase 5 reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class Phase5Inputs:
    """Versioned experiment reports consumed by the final report."""

    phase2_report: Path
    phase3_training_report: Path
    phase3_feature_report: Path
    phase4_validation_report: Path
    phase4_test_report: Path


@dataclass(frozen=True)
class Phase5Output:
    """Generated figure and summary locations."""

    figure_dir: Path
    summary_report: Path


@dataclass(frozen=True)
class Phase5Style:
    """Shared deterministic SVG dimensions."""

    width: int
    height: int


@dataclass(frozen=True)
class Phase5Config:
    """Complete Phase 5 reporting configuration."""

    inputs: Phase5Inputs
    output: Phase5Output
    style: Phase5Style


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def load_phase5_config(path: str | Path) -> Phase5Config:
    """Load and validate a Phase 5 YAML configuration."""
    raw = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "Phase 5 config")
    inputs = _mapping(raw.get("inputs"), "inputs")
    output = _mapping(raw.get("output"), "output")
    style = _mapping(raw.get("style"), "style")
    width = int(style.get("width", 960))
    height = int(style.get("height", 560))
    if width < 640 or height < 400:
        raise ValueError("Phase 5 SVG dimensions are too small")
    return Phase5Config(
        inputs=Phase5Inputs(
            phase2_report=Path(str(inputs["phase2_report"])),
            phase3_training_report=Path(str(inputs["phase3_training_report"])),
            phase3_feature_report=Path(str(inputs["phase3_feature_report"])),
            phase4_validation_report=Path(str(inputs["phase4_validation_report"])),
            phase4_test_report=Path(str(inputs["phase4_test_report"])),
        ),
        output=Phase5Output(
            figure_dir=Path(str(output["figure_dir"])),
            summary_report=Path(str(output["summary_report"])),
        ),
        style=Phase5Style(width=width, height=height),
    )
