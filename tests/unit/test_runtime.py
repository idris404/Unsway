"""Tests for deterministic runtime utilities."""

import random

import numpy as np
import torch

from unsway.runtime import resolve_device, seed_everything


def test_explicit_cpu_device() -> None:
    """CPU selection is always available."""
    assert resolve_device("cpu") == torch.device("cpu")


def test_seed_everything_is_reproducible() -> None:
    """All supported random number generators are reseeded."""
    seed_everything(7)
    first = (random.random(), np.random.rand(), torch.rand(1))
    seed_everything(7)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
