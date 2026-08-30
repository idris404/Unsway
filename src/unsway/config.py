"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]
DTypeName = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class ModelConfig:
    """Model loading options."""

    name: str
    device: DevicePreference = "auto"
    dtype: DTypeName = "float32"


@dataclass(frozen=True)
class ActivationConfig:
    """Target activation hook and its expected hidden width."""

    layer: int
    location: str = "hook_out"
    expected_width: int = 768

    @property
    def hook_name(self) -> str:
        """Return the canonical TransformerBridge hook name."""
        return f"blocks.{self.layer}.{self.location}"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for the Phase 0 smoke experiment."""

    seed: int
    prompt: str
    model: ModelConfig
    activation: ActivationConfig


def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return cast(dict[str, Any], value)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a Phase 0 YAML configuration file.

    Args:
        path: Path to the YAML configuration.

    Returns:
        A validated, immutable experiment configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required fields or supported values are invalid.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    experiment = _mapping(root.get("experiment"), "experiment")
    model = _mapping(root.get("model"), "model")
    activation = _mapping(root.get("activation"), "activation")

    try:
        device = str(model.get("device", "auto"))
        dtype = str(model.get("dtype", "float32"))
        if device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {dtype}")

        result = ExperimentConfig(
            seed=int(experiment["seed"]),
            prompt=str(experiment["prompt"]),
            model=ModelConfig(
                name=str(model["name"]),
                device=cast(DevicePreference, device),
                dtype=cast(DTypeName, dtype),
            ),
            activation=ActivationConfig(
                layer=int(activation["layer"]),
                location=str(activation.get("location", "hook_out")),
                expected_width=int(activation.get("expected_width", 768)),
            ),
        )
    except KeyError as error:
        raise ValueError(f"Missing required configuration field: {error.args[0]}") from error

    if not result.prompt.strip():
        raise ValueError("Prompt must not be empty")
    if result.activation.layer < 0:
        raise ValueError("Activation layer must be non-negative")
    if result.activation.expected_width <= 0:
        raise ValueError("Expected activation width must be positive")
    return result
