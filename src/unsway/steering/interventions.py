"""Construction and application of reproducible residual-stream directions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load_file
from torch.nn import functional as nn_functional

from unsway.data.source import sha256_file
from unsway.steering.config import Phase4Config


@dataclass(frozen=True)
class SteeringDirection:
    """A unit residual-stream direction with scientific provenance."""

    name: str
    kind: str
    index: int | None
    orientation: str
    values: torch.Tensor


class LastTokenSteeringHook:
    """Add a fixed-L2 direction at each sequence's final non-padding token."""

    def __init__(self, direction: torch.Tensor, strength: float, lengths: torch.Tensor) -> None:
        if direction.ndim != 1 or not torch.isfinite(direction).all():
            raise ValueError("Steering direction must be a finite vector")
        norm = torch.linalg.vector_norm(direction)
        if norm <= 0:
            raise ValueError("Steering direction must be non-zero")
        if lengths.ndim != 1 or (lengths <= 0).any():
            raise ValueError("Prompt lengths must be a positive vector")
        self.direction = direction.detach().to(torch.float32) / norm
        self.strength = float(strength)
        self.lengths = lengths.detach().to(torch.int64)

    def __call__(self, activation: torch.Tensor, hook: Any) -> torch.Tensor:
        """Return an intervened copy without mutating the cached activation."""
        del hook
        if activation.ndim != 3 or activation.shape[0] != self.lengths.numel():
            raise ValueError("Hook activation must align with prompt lengths")
        if activation.shape[-1] != self.direction.numel():
            raise ValueError("Steering direction width does not match the hook activation")
        result = activation.clone()
        rows = torch.arange(activation.shape[0], device=activation.device)
        positions = self.lengths.to(activation.device) - 1
        delta = self.direction.to(device=activation.device, dtype=activation.dtype) * self.strength
        result[rows, positions] += delta
        return result


def _orientation_sign(orientation: str) -> float:
    if orientation == "sycophancy_high":
        return -1.0
    if orientation == "resistance_high":
        return 1.0
    raise ValueError(f"Unsupported feature orientation: {orientation}")


def _feature_orientation(report: dict[str, Any], index: int) -> str:
    for feature in report["top_features"]:
        if int(feature["train"]["feature_index"]) == index:
            return str(feature["train"]["orientation"])
    raise ValueError(f"SAE feature {index} is absent from the Phase 3 feature report")


def build_directions(config: Phase4Config) -> tuple[SteeringDirection, ...]:
    """Validate Phase 3 artifacts and construct anti-sycophancy unit directions."""
    inputs = config.inputs
    checks = (
        (inputs.sae_path, inputs.sae_sha256, "SAE"),
        (inputs.training_report_path, inputs.training_report_sha256, "training report"),
        (inputs.feature_report_path, inputs.feature_report_sha256, "feature report"),
    )
    for path, expected, name in checks:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Phase 4 {name} checksum mismatch: {actual} != {expected}")

    training = json.loads(inputs.training_report_path.read_text(encoding="utf-8"))
    features = json.loads(inputs.feature_report_path.read_text(encoding="utf-8"))
    if training["artifact"]["sha256"] != inputs.sae_sha256:
        raise ValueError("Training report does not identify the configured SAE")
    if features["sae_artifact_sha256"] != inputs.sae_sha256:
        raise ValueError("Feature report does not identify the configured SAE")
    d_model = int(training["d_in"])
    tensors = load_file(inputs.sae_path)
    decoder = tensors["decoder_weight"].to(torch.float32)
    if decoder.ndim != 2 or decoder.shape[1] != d_model:
        raise ValueError("Invalid SAE decoder dimensions")

    raw_baseline = features["raw_neuron_baseline"]
    directions: list[SteeringDirection] = []
    for item in config.steering.interventions:
        if item.kind == "sae_feature":
            assert item.index is not None
            if item.index >= decoder.shape[0]:
                raise ValueError(f"SAE feature index {item.index} is out of bounds")
            orientation = _feature_orientation(features, item.index)
            values = decoder[item.index] * _orientation_sign(orientation)
        elif item.kind == "raw_neuron":
            assert item.index is not None
            if item.index >= d_model:
                raise ValueError(f"Raw neuron index {item.index} is out of bounds")
            if int(raw_baseline["dimension"]) != item.index:
                raise ValueError("Raw-neuron intervention does not match the Phase 3 baseline")
            orientation = str(raw_baseline["orientation"])
            values = torch.zeros(d_model, dtype=torch.float32)
            values[item.index] = _orientation_sign(orientation)
        else:
            assert item.seed is not None
            orientation = "random_control"
            generator = torch.Generator(device="cpu").manual_seed(item.seed)
            values = torch.randn(d_model, generator=generator)
        values = nn_functional.normalize(values, dim=0)
        directions.append(
            SteeringDirection(
                name=item.name,
                kind=item.kind,
                index=item.index,
                orientation=orientation,
                values=values,
            )
        )
    return tuple(directions)
