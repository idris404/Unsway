"""Determinism and guardrail tests for Phase 6D."""

from __future__ import annotations

import torch

from unsway.phase6.directions import _caa_cv, _stratified_folds, load_phase6d_methods
from unsway.phase6.validation import _select_candidate


def test_phase6d_methods_are_frozen_and_hashable() -> None:
    """The committed method specification has a stable SHA-256 identity."""
    methods = load_phase6d_methods("configs/phase6d.yaml")
    assert len(methods["methods_sha256"]) == 64
    assert methods["selection"]["primary_metric"] == "steered_minus_baseline_pressure_effect"
    assert methods["construction"]["selected_caa_layers"] == 2


def test_stratified_folds_are_deterministic_and_balanced() -> None:
    """Every binary class is distributed across all train-only folds."""
    labels = torch.tensor([0] * 10 + [1] * 15)
    first = _stratified_folds(labels, 5, 606)
    second = _stratified_folds(labels, 5, 606)
    assert torch.equal(first, second)
    for label in (0, 1):
        counts = torch.bincount(first[labels == label], minlength=5)
        assert int(counts.max() - counts.min()) <= 1


def test_caa_cv_detects_a_separable_resistance_direction() -> None:
    """Cross-validated CAA scoring rewards a stable behavior axis."""
    generator = torch.Generator().manual_seed(7)
    resistant = torch.randn(40, 4, generator=generator) * 0.05
    sycophantic = torch.randn(40, 4, generator=generator) * 0.05
    resistant[:, 0] += 2
    sycophantic[:, 0] -= 2
    values = torch.cat([resistant, sycophantic])
    labels = torch.tensor([0] * 40 + [1] * 40)
    assert _caa_cv(values, labels, 5, 606) > 0.99


def _condition(
    name: str,
    effect_delta: float,
    accuracy_delta: float,
    source_deltas: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "intervention": name,
        "family": "behavioral_caa",
        "hook_name": "blocks.5.hook_out",
        "strength": 2.0,
        "delta": {
            "pressure_effect": effect_delta,
            "initial_accuracy": accuracy_delta,
        },
        "source_pressure_effect_deltas": dict(
            zip(("arc", "csqa", "obqa"), source_deltas, strict=True)
        ),
    }


def test_candidate_selection_enforces_all_frozen_guardrails() -> None:
    """Effect, accuracy, and cross-source constraints must pass together."""
    too_small = _condition("small", -0.019, 0.0, (-0.1, -0.1, -0.1))
    too_costly = _condition("costly", -0.04, -0.021, (-0.1, -0.1, -0.1))
    one_source = _condition("narrow", -0.04, 0.0, (-0.1, 0.0, 0.1))
    winner = _condition("winner", -0.03, -0.01, (-0.02, -0.01, 0.01))
    selected = _select_candidate(
        [too_small, too_costly, one_source, winner],
        min_effect=0.02,
        max_accuracy_drop=0.02,
        min_sources=2,
    )
    assert selected is not None
    assert selected["intervention"] == "winner"


def test_candidate_selection_returns_none_when_validation_fails() -> None:
    """The unopened confirmatory test stays blocked after a negative validation."""
    selected = _select_candidate(
        [_condition("failed", -0.01, 0.0, (-0.01, -0.01, -0.01))],
        min_effect=0.02,
        max_accuracy_drop=0.02,
        min_sources=2,
    )
    assert selected is None
