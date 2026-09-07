"""Tests for safe residual-stream interventions."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from unsway.data.io import write_manifest
from unsway.data.source import sha256_file
from unsway.steering.config import InterventionConfig, load_phase4_config
from unsway.steering.interventions import LastTokenSteeringHook, build_directions


def test_last_token_hook_changes_only_unpadded_final_positions() -> None:
    """Each batch row is steered at its own final non-padding token."""
    activation = torch.zeros((2, 4, 3))
    hook = LastTokenSteeringHook(torch.tensor([3.0, 0.0, 0.0]), 2.0, torch.tensor([2, 4]))

    result = hook(activation, None)

    assert result[0, 1].tolist() == [2.0, 0.0, 0.0]
    assert result[1, 3].tolist() == [2.0, 0.0, 0.0]
    assert torch.count_nonzero(result) == 2
    assert torch.count_nonzero(activation) == 0


def test_last_token_hook_rejects_wrong_width() -> None:
    """Silent broadcasting cannot corrupt an intervention."""
    hook = LastTokenSteeringHook(torch.ones(2), 1.0, torch.tensor([1]))

    with pytest.raises(ValueError, match="width"):
        hook(torch.zeros((1, 1, 3)), None)


def test_build_directions_applies_phase3_orientations(tmp_path: Path) -> None:
    """Correlational orientation is converted to an anti-sycophancy causal sign."""
    sae_path = tmp_path / "sae.safetensors"
    save_file({"decoder_weight": torch.eye(3)}, sae_path)
    sae_sha = sha256_file(sae_path)
    training_path = tmp_path / "training.json"
    write_manifest(training_path, {"d_in": 3, "artifact": {"sha256": sae_sha}})
    feature_path = tmp_path / "features.json"
    write_manifest(
        feature_path,
        {
            "sae_artifact_sha256": sae_sha,
            "top_features": [{"train": {"feature_index": 1, "orientation": "sycophancy_high"}}],
            "raw_neuron_baseline": {"dimension": 2, "orientation": "resistance_high"},
        },
    )
    base = load_phase4_config("configs/phase4_smoke.yaml")
    inputs = replace(
        base.inputs,
        sae_path=sae_path,
        sae_sha256=sae_sha,
        training_report_path=training_path,
        training_report_sha256=sha256_file(training_path),
        feature_report_path=feature_path,
        feature_report_sha256=sha256_file(feature_path),
    )
    steering = replace(
        base.steering,
        interventions=(
            InterventionConfig("sae", "sae_feature", 1, None, None),
            InterventionConfig("neuron", "raw_neuron", 2, None, None),
            InterventionConfig("random", "random_direction", None, 7, "sae"),
        ),
    )

    directions = build_directions(replace(base, inputs=inputs, steering=steering))

    assert directions[0].values.tolist() == [0.0, -1.0, 0.0]
    assert directions[1].values.tolist() == [0.0, 0.0, 1.0]
    assert torch.linalg.vector_norm(directions[2].values).item() == pytest.approx(1.0)
