"""Runtime utilities for device selection and reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch

from unsway.config import DevicePreference, DTypeName


def resolve_device(preference: DevicePreference = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch device.

    Automatic selection prefers CUDA, then Apple Metal, then CPU.

    Args:
        preference: Requested device or ``auto``.

    Returns:
        An available PyTorch device.

    Raises:
        RuntimeError: If an explicitly requested accelerator is unavailable.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(preference)


def resolve_dtype(name: DTypeName) -> torch.dtype:
    """Map a configuration dtype name to a PyTorch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
