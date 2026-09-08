"""Tests for the frozen Phase 6 protocol."""

from dataclasses import replace
from pathlib import Path

import pytest

from unsway.phase6.config import load_phase6_config
from unsway.phase6.dataset import write_protocol_manifest


def test_phase6_production_protocol_has_fresh_holdout_guardrails() -> None:
    """The production config freezes the planned sources and success criteria."""
    config = load_phase6_config("configs/phase6.yaml")

    assert {artifact.source for artifact in config.artifacts} == {
        "ai2_arc",
        "commonsense_qa",
        "openbookqa",
    }
    assert len(config.protocol.hook_points) == 12
    assert config.protocol.min_test_eligible == 300
    assert config.protocol.min_effect_pp == 2.0
    assert config.protocol.max_accuracy_drop_pp == 2.0
    assert config.protocol.refuse_test_overwrite is True


def test_protocol_manifest_rejects_silent_mutation(tmp_path: Path) -> None:
    """Changing a frozen scientific choice requires a new experiment identity."""
    config = load_phase6_config("configs/phase6.yaml")
    config = replace(
        config,
        protocol=replace(config.protocol, manifest_path=tmp_path / "protocol.json"),
    )
    write_protocol_manifest(config)
    changed = replace(
        config,
        protocol=replace(config.protocol, min_effect_pp=1.0),
    )

    with pytest.raises(ValueError, match="differs from the existing pre-registration"):
        write_protocol_manifest(changed)
