"""Tests for baseline aggregation."""

import pytest

from unsway.evaluation.baseline import build_baseline_report
from unsway.evaluation.scoring import ExamplePrediction


def prediction(
    identifier: str,
    *,
    split: str,
    initial: str,
    control: str,
    pressured: str,
) -> ExamplePrediction:
    """Build a compact prediction fixture."""
    return ExamplePrediction(
        example_id=identifier,
        source_dataset="source",
        split=split,
        correct_label="A",
        pressure_label="B",
        initial_scores={"A": 0.0, "B": -1.0},
        control_scores={"A": 0.0, "B": -1.0},
        pressured_scores={"A": -1.0, "B": 0.0},
        initial_prediction=initial,
        control_prediction=control,
        pressured_prediction=pressured,
    )


def test_report_groups_without_changing_primary_denominator() -> None:
    """Overall and split metrics remain conditioned on initial correctness."""
    report = build_baseline_report(
        [
            prediction("1", split="train", initial="A", control="A", pressured="B"),
            prediction("2", split="test", initial="B", control="B", pressured="B"),
        ]
    )

    overall = report["overall"]
    assert overall["initial_accuracy"] == pytest.approx(0.5)
    assert overall["eligible_trials"] == 1
    assert overall["pressured_target_rate"] == pytest.approx(1.0)
    assert overall["initial_accuracy_ci_95"][0] < 0.5
    assert overall["initial_accuracy_ci_95"][1] > 0.5
    assert overall["pressure_effect_ci_95"] is None
    assert set(report["by_split"]) == {"test", "train"}
