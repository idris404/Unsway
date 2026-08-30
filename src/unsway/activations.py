"""Safe extraction and validation of transformer activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class CacheableModel(Protocol):
    """Minimal interface required from a TransformerLens model."""

    def run_with_cache(self, prompt: str, *, names_filter: list[str]) -> tuple[Any, Any]:
        """Run a forward pass and return its output and activation cache."""
        ...


@dataclass(frozen=True)
class ActivationBatch:
    """A validated activation tensor and its provenance."""

    hook_name: str
    prompt: str
    values: torch.Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the activation shape as plain integers."""
        return tuple(self.values.shape)


def extract_activation(
    model: CacheableModel,
    prompt: str,
    hook_name: str,
    *,
    expected_width: int | None = None,
) -> ActivationBatch:
    """Extract one named activation from one prompt and validate its shape.

    Only the requested hook is cached, avoiding the large memory overhead of a
    complete activation cache.

    Args:
        model: A TransformerLens-compatible model.
        prompt: Non-empty text to tokenize and process.
        hook_name: Canonical TransformerLens hook name.
        expected_width: Optional expected final tensor dimension.

    Returns:
        A detached CPU tensor together with its hook and prompt metadata.

    Raises:
        ValueError: If inputs or tensor contents are invalid.
        KeyError: If the requested hook is absent from the returned cache.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if not hook_name:
        raise ValueError("Hook name must not be empty")

    with torch.inference_mode():
        _, cache = model.run_with_cache(prompt, names_filter=[hook_name])

    if hook_name not in cache:
        available = ", ".join(sorted(str(key) for key in cache))
        raise KeyError(f"Hook '{hook_name}' was not cached. Available hooks: {available or 'none'}")

    values = cache[hook_name]
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"Hook '{hook_name}' returned {type(values).__name__}, expected Tensor")
    if values.ndim != 3:
        raise ValueError(
            f"Hook '{hook_name}' has shape {tuple(values.shape)}; expected [batch, tokens, width]"
        )
    if expected_width is not None and values.shape[-1] != expected_width:
        raise ValueError(
            f"Hook '{hook_name}' width is {values.shape[-1]}, expected {expected_width}"
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"Hook '{hook_name}' contains non-finite values")

    return ActivationBatch(
        hook_name=hook_name,
        prompt=prompt,
        values=values.detach().to(device="cpu").clone(),
    )
