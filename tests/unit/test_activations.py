"""Tests for activation extraction invariants."""

from typing import Any

import pytest
import torch

from unsway.activations import extract_activation


class FakeCacheableModel:
    """Small deterministic stand-in for TransformerBridge."""

    def __init__(self, activation: object) -> None:
        self.activation = activation
        self.requested_hooks: list[str] = []

    def run_with_cache(
        self, prompt: str, *, names_filter: list[str]
    ) -> tuple[None, dict[str, Any]]:
        self.requested_hooks = names_filter
        return None, {names_filter[0]: self.activation}


def test_extracts_only_requested_hook_and_moves_to_cpu() -> None:
    """Extraction caches only its target and returns a safe detached copy."""
    source = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).requires_grad_()
    model = FakeCacheableModel(source)

    result = extract_activation(
        model,
        "A test prompt",
        "blocks.0.hook_out",
        expected_width=8,
    )

    assert model.requested_hooks == ["blocks.0.hook_out"]
    assert result.shape == (1, 3, 8)
    assert result.values.device.type == "cpu"
    assert not result.values.requires_grad
    assert result.values.data_ptr() != source.data_ptr()


@pytest.mark.parametrize(
    ("activation", "message"),
    [
        (torch.ones(3, 8), "expected \\[batch, tokens, width\\]"),
        (torch.full((1, 3, 8), torch.nan), "non-finite"),
        (torch.ones(1, 3, 7), "width is 7, expected 8"),
    ],
)
def test_rejects_invalid_activations(activation: torch.Tensor, message: str) -> None:
    """Shape, width, and finite-value errors cannot pass silently."""
    model = FakeCacheableModel(activation)

    with pytest.raises(ValueError, match=message):
        extract_activation(model, "prompt", "blocks.0.hook_out", expected_width=8)


def test_rejects_missing_hook() -> None:
    """A missing hook produces an actionable error."""

    class MissingHookModel:
        def run_with_cache(
            self, prompt: str, *, names_filter: list[str]
        ) -> tuple[None, dict[str, torch.Tensor]]:
            return None, {}

    with pytest.raises(KeyError, match="was not cached"):
        extract_activation(MissingHookModel(), "prompt", "blocks.0.hook_out")
