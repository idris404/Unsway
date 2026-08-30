"""TransformerLens model loading."""

from __future__ import annotations

from typing import Any

from unsway.config import ModelConfig
from unsway.runtime import resolve_device, resolve_dtype


def load_transformer(config: ModelConfig) -> Any:
    """Load a Hugging Face model through TransformerLens' supported bridge API.

    Args:
        config: Model identifier, device preference, and numerical precision.

    Returns:
        An evaluation-mode ``TransformerBridge`` instance.
    """
    from transformer_lens.model_bridge import TransformerBridge

    device = resolve_device(config.device)
    model = TransformerBridge.boot_transformers(
        config.name,
        device=str(device),
        dtype=resolve_dtype(config.dtype),
    )
    model.eval()
    return model
