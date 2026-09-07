"""Tests for post-hoc feature statistics."""

import numpy as np
import pytest

from unsway.features.analysis import binary_auroc


def test_binary_auroc_perfect_and_reversed_rankings() -> None:
    """AUROC has the expected orientation for sycophancy-high features."""
    labels = np.array([0, 0, 1, 1])

    assert binary_auroc(np.array([0.0, 0.1, 0.9, 1.0]), labels) == pytest.approx(1.0)
    assert binary_auroc(np.array([1.0, 0.9, 0.1, 0.0]), labels) == pytest.approx(0.0)


def test_binary_auroc_handles_ties() -> None:
    """A constant feature has chance-level AUROC."""
    assert binary_auroc(np.ones(4), np.array([0, 1, 0, 1])) == pytest.approx(0.5)
