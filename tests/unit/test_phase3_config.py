"""Tests for Phase 3 configuration."""

from pathlib import Path

import pytest

from unsway.features.config import load_phase3_config


def test_load_full_and_smoke_configs() -> None:
    """Both cloud-scale and local smoke configurations are valid."""
    full = load_phase3_config("configs/phase3.yaml")
    smoke = load_phase3_config("configs/phase3_smoke.yaml")

    assert full.extraction.hook_name == "blocks.5.hook_out"
    assert full.sae.expansion_factor == 8
    assert smoke.extraction.max_examples_per_split == 24
    assert smoke.output.artifact_dir.parts[0] == "artifacts"


def test_rejects_non_positive_sae_width(tmp_path: Path) -> None:
    """Invalid SAE dimensions fail before extraction."""
    source = Path("configs/phase3_smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("expansion_factor: 2", "expansion_factor: 0"), encoding="utf-8")

    with pytest.raises(ValueError, match="must be positive"):
        load_phase3_config(path)
