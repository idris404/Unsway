"""Leakage-safe Phase 6D validation sweep and confirmatory selection."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from unsway.data.io import load_dataset, write_jsonl_records, write_manifest
from unsway.data.source import sha256_file
from unsway.evaluation.scoring import ExamplePrediction, dynamic_batches
from unsway.features.corpus import load_prediction_map
from unsway.phase6.directions import load_phase6d_methods
from unsway.steering.evaluation import (
    SteerableModel,
    condition_report,
    score_steered_batch,
    steering_metrics,
)

Progress = Callable[[str, float, int, int], None]


def _score(
    model: SteerableModel,
    examples: list[Any],
    hook_name: str,
    direction: torch.Tensor,
    strength: float,
    max_batch_size: int,
    max_batch_tokens: int,
    progress: Progress | None,
    name: str,
) -> list[ExamplePrediction]:
    batches = list(
        dynamic_batches(
            examples,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
        )
    )
    predictions: list[ExamplePrediction] = []
    for index, batch in enumerate(batches, start=1):
        predictions.extend(score_steered_batch(model, batch, hook_name, direction, strength))
        if progress is not None:
            progress(name, strength, index, len(batches))
    predictions.sort(key=lambda item: item.example_id)
    return predictions


def _source_deltas(
    baseline: list[ExamplePrediction], candidate: list[ExamplePrediction]
) -> dict[str, float]:
    sources = sorted({item.source_dataset for item in baseline})
    output: dict[str, float] = {}
    for source in sources:
        base = [item for item in baseline if item.source_dataset == source]
        selected_ids = {item.example_id for item in base}
        steered = [item for item in candidate if item.example_id in selected_ids]
        output[source] = float(condition_report(base, steered)["delta"]["pressure_effect"])
    return output


def _select_candidate(
    conditions: list[dict[str, Any]],
    *,
    min_effect: float,
    max_accuracy_drop: float,
    min_sources: int,
) -> dict[str, Any] | None:
    feasible = []
    for row in conditions:
        delta = row["delta"]
        source_improvements = sum(
            float(value) < 0 for value in row["source_pressure_effect_deltas"].values()
        )
        accuracy_drop = max(0.0, -float(delta["initial_accuracy"]))
        if (
            float(delta["pressure_effect"]) <= -min_effect
            and accuracy_drop <= max_accuracy_drop
            and source_improvements >= min_sources
        ):
            feasible.append((row, source_improvements, accuracy_drop))
    if not feasible:
        return None
    row, source_improvements, accuracy_drop = min(
        feasible,
        key=lambda item: (
            float(item[0]["delta"]["pressure_effect"]),
            item[2],
            abs(float(item[0]["strength"])),
            str(item[0]["intervention"]),
        ),
    )
    return {
        "intervention": row["intervention"],
        "family": row["family"],
        "hook_name": row["hook_name"],
        "strength": row["strength"],
        "pressure_effect_delta": row["delta"]["pressure_effect"],
        "initial_accuracy_delta": row["delta"]["initial_accuracy"],
        "sources_improved": source_improvements,
    }


def run_phase6d_validation(
    methods_path: str | Path,
    model: SteerableModel,
    *,
    max_batch_size: int,
    max_batch_tokens: int,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Evaluate positive doses on validation and freeze at most one real candidate."""
    methods = load_phase6d_methods(methods_path)
    inputs = methods["inputs"]
    output = methods["output"]
    selection = methods["selection"]
    directions_report_path = Path(str(output["directions_report_path"]))
    directions_report = json.loads(directions_report_path.read_text(encoding="utf-8"))
    if directions_report.get("status") != "train_only_directions_frozen":
        raise ValueError("Phase 6D directions are not frozen")
    if directions_report.get("methods_sha256") != methods["methods_sha256"]:
        raise ValueError("Phase 6D direction methods mismatch")
    directions_path = Path(str(output["directions_path"]))
    if sha256_file(directions_path) != directions_report["artifact"]["sha256"]:
        raise ValueError("Phase 6D direction artifact checksum mismatch")
    tensors = load_file(directions_path)

    examples = sorted(
        (
            item
            for item in load_dataset("data/processed/phase6_sycophancy.jsonl")
            if item.split.value == "validation"
        ),
        key=lambda item: item.example_id,
    )
    prediction_map = load_prediction_map(Path(str(inputs["baseline_predictions_path"])))
    baseline = [prediction_map[item.example_id] for item in examples]
    baseline_metrics = steering_metrics(baseline, baseline)
    rows_by_name = {str(row["name"]): row for row in directions_report["directions"]}
    real_rows = [
        row for row in directions_report["directions"] if row["family"] != "matched_random"
    ]
    conditions: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for direction_row in real_rows:
        name = str(direction_row["name"])
        direction = tensors[str(direction_row["tensor_key"])]
        for raw_strength in selection["strengths"]:
            strength = float(raw_strength)
            predictions = _score(
                model,
                examples,
                str(direction_row["hook_name"]),
                direction,
                strength,
                max_batch_size,
                max_batch_tokens,
                progress,
                name,
            )
            summary = condition_report(baseline, predictions)
            conditions.append(
                {
                    "intervention": name,
                    "family": direction_row["family"],
                    "hook_name": direction_row["hook_name"],
                    "strength": strength,
                    "source_pressure_effect_deltas": _source_deltas(baseline, predictions),
                    **summary,
                }
            )
            prediction_rows.extend(
                {"intervention": name, "strength": strength, **item.to_dict()}
                for item in predictions
            )

    winner = _select_candidate(
        conditions,
        min_effect=float(selection["min_effect_pp"]) / 100,
        max_accuracy_drop=float(selection["max_accuracy_drop_pp"]) / 100,
        min_sources=int(selection["min_sources_consistent"]),
    )
    random_control: dict[str, Any] | None = None
    if winner is not None:
        random_name = f"matched_random__{winner['intervention']}"
        random_row = rows_by_name[random_name]
        predictions = _score(
            model,
            examples,
            str(random_row["hook_name"]),
            tensors[str(random_row["tensor_key"])],
            float(winner["strength"]),
            max_batch_size,
            max_batch_tokens,
            progress,
            random_name,
        )
        summary = condition_report(baseline, predictions)
        random_control = {
            "intervention": random_name,
            "family": "matched_random",
            "hook_name": random_row["hook_name"],
            "strength": winner["strength"],
            "source_pressure_effect_deltas": _source_deltas(baseline, predictions),
            **summary,
        }
        conditions.append(random_control)
        prediction_rows.extend(
            {"intervention": random_name, "strength": winner["strength"], **item.to_dict()}
            for item in predictions
        )

    predictions_path = Path(str(output["validation_predictions_path"]))
    write_jsonl_records(predictions_path, prediction_rows)
    report = {
        "schema_version": 1,
        "status": "confirmatory_candidate_frozen"
        if winner is not None
        else "no_candidate_passed_validation",
        "methods_sha256": methods["methods_sha256"],
        "directions_report_sha256": sha256_file(directions_report_path),
        "directions_artifact_sha256": sha256_file(directions_path),
        "split": "validation",
        "examples": len(examples),
        "baseline": baseline_metrics,
        "selection_guardrails": selection,
        "conditions": conditions,
        "selected_confirmatory_candidate": winner,
        "matched_random_control": random_control,
        "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
        "test_prompts_scored": False,
    }
    write_manifest(Path(str(output["validation_report_path"])), report)
    return report


__all__ = ["_select_candidate", "run_phase6d_validation"]
