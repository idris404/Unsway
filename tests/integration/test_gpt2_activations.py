"""Integration test against real GPT-2 weights and TransformerLens hooks."""

import pytest

from unsway.activations import extract_activation
from unsway.config import ModelConfig
from unsway.model import load_transformer


@pytest.mark.integration
def test_gpt2_small_residual_activation_shape() -> None:
    """GPT-2 small exposes a finite 768-wide layer-5 residual activation."""
    model = load_transformer(
        ModelConfig(name="openai-community/gpt2", device="cpu", dtype="float32")
    )

    activation = extract_activation(
        model,
        "The capital of France is",
        "blocks.5.hook_out",
        expected_width=768,
    )

    assert activation.shape[0] == 1
    assert activation.shape[1] > 0
    assert activation.shape[2] == 768
