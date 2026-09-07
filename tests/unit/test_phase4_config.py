"""Tests for Phase 4 experiment configuration."""

from pathlib import Path

import pytest

from unsway.steering.config import load_phase4_config


def test_load_full_and_smoke_phase4_configs() -> None:
    """Both steering configurations encode the frozen causal protocol."""
    full = load_phase4_config("configs/phase4.yaml")
    smoke = load_phase4_config("configs/phase4_smoke.yaml")

    assert full.steering.hook_name == "blocks.5.hook_out"
    assert full.steering.interventions[0].index == 4825
    assert full.steering.interventions[2].match_strength_to == "sae_4825"
    assert smoke.steering.max_examples_per_split == 24


def test_phase4_config_requires_zero_strength(tmp_path: Path) -> None:
    """A baseline dose is mandatory for an interpretable sweep."""
    source = Path("configs/phase4_smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("[-1.0, 0.0, 1.0]", "[-1.0, 1.0]"), encoding="utf-8")

    with pytest.raises(ValueError, match="include zero"):
        load_phase4_config(path)
