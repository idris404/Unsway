"""Tests for Phase 4 fixed-population metrics and selection."""

import pytest
import torch

from unsway.evaluation.scoring import ExamplePrediction
from unsway.steering.evaluation import (
    choose_validation_strength,
    condition_report,
    score_prompts_with_steering,
    steering_metrics,
)


class FakeTokenizer:
    """Map answer letters to stable token identifiers."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [{"A": 1, "B": 2}[text]]


class FakeSteerableModel:
    """Expose logits controlled by the first hooked residual dimension."""

    tokenizer = FakeTokenizer()

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        texts = [text] if isinstance(text, str) else text
        return torch.zeros((len(texts), max(len(item) for item in texts)), dtype=torch.long)

    def run_with_hooks(
        self,
        input: torch.Tensor,
        *,
        fwd_hooks: list[tuple[str, object]],
    ) -> torch.Tensor:
        activation = torch.zeros((*input.shape, 3))
        hook = fwd_hooks[0][1]
        assert callable(hook)
        activation = hook(activation, None)
        logits = torch.zeros((*input.shape, 4))
        logits[..., 1] = activation[..., 0]
        logits[..., 2] = -activation[..., 0]
        return logits


def prediction(
    identifier: str,
    *,
    initial: str = "A",
    control: str = "A",
    pressured: str = "B",
) -> ExamplePrediction:
    """Create a compact intervention outcome."""
    return ExamplePrediction(
        example_id=identifier,
        source_dataset="source",
        split="validation",
        correct_label="A",
        pressure_label="B",
        initial_scores={"A": 0.0, "B": -1.0},
        control_scores={"A": 0.0, "B": -1.0},
        pressured_scores={"A": -1.0, "B": 0.0},
        initial_prediction=initial,
        control_prediction=control,
        pressured_prediction=pressured,
    )


def test_steered_scoring_changes_candidate_preference() -> None:
    """The hook's residual delta reaches next-token candidate scoring."""
    scores = score_prompts_with_steering(
        FakeSteerableModel(),
        ["xx", "xxxx"],
        [("A", "B"), ("A", "B")],
        "blocks.5.hook_out",
        torch.tensor([1.0, 0.0, 0.0]),
        2.0,
    )

    assert all(row["A"] > row["B"] for row in scores)


def test_metrics_keep_phase2_eligibility_fixed() -> None:
    """Steering cannot improve its denominator by changing initial answers."""
    baseline = [prediction("1"), prediction("2", initial="B")]
    steered = [prediction("1", initial="B", pressured="A"), prediction("2", initial="A")]

    metrics = steering_metrics(baseline, steered)
    report = condition_report(baseline, steered)

    assert metrics["fixed_eligible_trials"] == 1
    assert metrics["pressured_target_rate"] == 0.0
    assert metrics["initial_accuracy"] == pytest.approx(0.5)
    assert report["delta"]["pressured_target_rate"] == -1.0


def test_validation_selection_enforces_accuracy_guardrail() -> None:
    """Selection ignores harmful doses and chooses the best smallest safe dose."""
    baseline = {
        "initial_accuracy": 0.8,
        "pressured_target_rate": 0.4,
        "pressure_effect": 0.3,
    }
    conditions = [
        {
            "strength": 1.0,
            "metrics": {
                "initial_accuracy": 0.79,
                "pressured_target_rate": 0.3,
                "pressure_effect": 0.2,
            },
        },
        {
            "strength": 2.0,
            "metrics": {
                "initial_accuracy": 0.78,
                "pressured_target_rate": 0.2,
                "pressure_effect": 0.1,
            },
        },
        {
            "strength": 4.0,
            "metrics": {
                "initial_accuracy": 0.60,
                "pressured_target_rate": 0.0,
                "pressure_effect": 0.0,
            },
        },
        {
            "strength": -2.0,
            "metrics": {
                "initial_accuracy": 0.8,
                "pressured_target_rate": 0.0,
                "pressure_effect": -0.2,
            },
        },
    ]

    assert choose_validation_strength(baseline, conditions, 0.05) == 2.0
