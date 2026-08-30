"""Tests for typed YAML configuration."""

from pathlib import Path

import pytest

from unsway.config import load_config


def test_load_phase0_config() -> None:
    """The committed Phase 0 configuration is valid and uses a canonical hook."""
    config = load_config(Path("configs/phase0.yaml"))

    assert config.model.name == "openai-community/gpt2"
    assert config.activation.hook_name == "blocks.5.hook_out"
    assert config.activation.expected_width == 768


def test_rejects_invalid_device(tmp_path: Path) -> None:
    """Invalid device names fail before model loading."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
experiment: {seed: 42, prompt: hello}
model: {name: gpt2, device: tpu}
activation: {layer: 0}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported device"):
        load_config(config_path)
