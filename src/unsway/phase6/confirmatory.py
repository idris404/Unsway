"""One-shot confirmatory evaluation on the unopened Phase 6C holdout."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from safetensors.torch import load_file

from unsway.data.io import load_dataset, write_jsonl_records, write_manifest
from unsway.data.source import sha256_file
from unsway.evaluation.scoring import ExamplePrediction, dynamic_batches, score_batch
from unsway.phase6.config import Phase6Config
from unsway.phase6.validation import _score, _source_deltas
from unsway.steering.evaluation import SteerableModel, condition_report, steering_metrics

Progress = Callable[[str, float, int, int], None]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def load_phase6e_config(path: str | Path) -> dict[str, Any]:
    """Load and hash the confirmatory specification without path-dependent fields."""
    raw = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "Phase 6E config")
    for section in ("inputs", "candidate", "success", "output"):
        _mapping(raw.get(section), section)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported Phase 6E schema")
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    raw["config_sha256"] = hashlib.sha256(encoded).hexdigest()
    return raw


def freeze_phase6e_candidate(path: str | Path) -> dict[str, Any]:
    """Bind the predeclared candidate to the completed validation artifacts."""
    config = load_phase6e_config(path)
    inputs = config["inputs"]
    output = config["output"]
    validation_path = Path(str(inputs["validation_report_path"]))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "confirmatory_candidate_frozen":
        raise ValueError("Phase 6D did not freeze a confirmatory candidate")
    if validation.get("test_prompts_scored") is not False:
        raise ValueError("Phase 6E requires a still-unopened test holdout")
    if validation.get("methods_sha256") != inputs["phase6d_methods_sha256"]:
        raise ValueError("Phase 6D methods checksum mismatch")
    candidate = validation.get("selected_confirmatory_candidate")
    expected = config["candidate"]
    for key in ("intervention", "family", "hook_name", "strength"):
        if candidate.get(key) != expected[key]:
            raise ValueError(f"Frozen candidate mismatch for {key}")
    directions_path = Path(str(inputs["directions_path"]))
    directions_report_path = Path(str(inputs["directions_report_path"]))
    directions_report = json.loads(directions_report_path.read_text(encoding="utf-8"))
    if sha256_file(directions_path) != directions_report["artifact"]["sha256"]:
        raise ValueError("Phase 6D directions artifact checksum mismatch")
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_confirmatory_test",
        "config_sha256": config["config_sha256"],
        "phase6d_validation_sha256": sha256_file(validation_path),
        "phase6d_directions_report_sha256": sha256_file(directions_report_path),
        "phase6d_directions_sha256": sha256_file(directions_path),
        "candidate": expected,
        "success": config["success"],
        "test_prompts_scored": False,
    }
    write_manifest(Path(str(output["freeze_manifest_path"])), manifest)
    return manifest


def _initial_map(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


def _score_baseline(
    model: Any,
    examples: list[Any],
    max_batch_size: int,
    max_batch_tokens: int,
) -> list[ExamplePrediction]:
    predictions: list[ExamplePrediction] = []
    for batch in dynamic_batches(
        examples, max_batch_size=max_batch_size, max_batch_tokens=max_batch_tokens
    ):
        predictions.extend(score_batch(model, batch))
    return sorted(predictions, key=lambda item: item.example_id)


def _bootstrap_interval(
    baseline: list[ExamplePrediction],
    candidate: list[ExamplePrediction],
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Return deterministic source-stratified paired bootstrap intervals."""
    base = {row.example_id: row for row in baseline}
    steered = {row.example_id: row for row in candidate}
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in baseline:
        grouped[row.source_dataset].append(row.example_id)
    generator = np.random.default_rng(seed)
    effect_values: list[float] = []
    accuracy_values: list[float] = []
    for _ in range(replicates):
        sampled = [
            identifier
            for source in sorted(grouped)
            for identifier in generator.choice(grouped[source], len(grouped[source]), replace=True)
        ]
        eligible = [
            identifier
            for identifier in sampled
            if base[identifier].initial_prediction == base[identifier].correct_label
        ]
        effect = [
            (
                int(steered[item].pressured_prediction == steered[item].pressure_label)
                - int(steered[item].control_prediction == steered[item].pressure_label)
            )
            - (
                int(base[item].pressured_prediction == base[item].pressure_label)
                - int(base[item].control_prediction == base[item].pressure_label)
            )
            for item in eligible
        ]
        accuracy = [
            int(steered[item].initial_prediction == steered[item].correct_label)
            - int(base[item].initial_prediction == base[item].correct_label)
            for item in sampled
        ]
        effect_values.append(float(np.mean(effect)))
        accuracy_values.append(float(np.mean(accuracy)))
    return {
        "pressure_effect_delta_ci_95": [
            float(value) for value in np.quantile(effect_values, [0.025, 0.975])
        ],
        "initial_accuracy_delta_ci_95": [
            float(value) for value in np.quantile(accuracy_values, [0.025, 0.975])
        ],
    }


def run_phase6e_test(
    path: str | Path,
    holdout: Phase6Config,
    model: SteerableModel,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Open the replacement holdout once for the frozen candidate and its control."""
    config = load_phase6e_config(path)
    inputs = config["inputs"]
    output = config["output"]
    report_path = Path(str(output["test_report_path"]))
    predictions_path = Path(str(output["test_predictions_path"]))
    if holdout.protocol.refuse_test_overwrite and (
        report_path.exists() or predictions_path.exists()
    ):
        raise FileExistsError("Refusing to overwrite frozen Phase 6E test outputs")
    freeze_path = Path(str(output["freeze_manifest_path"]))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_confirmatory_test":
        raise ValueError("Phase 6E candidate freeze is missing")
    if freeze.get("config_sha256") != config["config_sha256"]:
        raise ValueError("Phase 6E config changed after candidate freeze")
    validation_path = Path(str(inputs["validation_report_path"]))
    if freeze.get("phase6d_validation_sha256") != sha256_file(validation_path):
        raise ValueError("Phase 6D validation changed after candidate freeze")
    baseline_report = json.loads(
        holdout.phase6b_output.baseline_report_path.read_text(encoding="utf-8")
    )
    initial = baseline_report["test_initial_only"]
    if (
        baseline_report.get("status") != "ready_for_frozen_test"
        or initial.get("pressure_scored") is not False
        or initial.get("control_scored") is not False
    ):
        raise ValueError("Replacement holdout was not safely gated")
    examples = sorted(load_dataset(holdout.dataset.dataset_path), key=lambda row: row.example_id)
    if any(row.split.value != "test" for row in examples):
        raise ValueError("Phase 6E accepts only the replacement test split")
    baseline = _score_baseline(
        model,
        examples,
        holdout.runtime.max_batch_size,
        holdout.runtime.max_batch_tokens,
    )
    frozen_initial = _initial_map(holdout.phase6b_output.test_initial_predictions_path)
    if any(
        frozen_initial[row.example_id]["initial_prediction"] != row.initial_prediction
        for row in baseline
    ):
        raise ValueError("Confirmatory baseline does not reproduce frozen initial decisions")
    directions_report = json.loads(
        Path(str(inputs["directions_report_path"])).read_text(encoding="utf-8")
    )
    directions = load_file(str(inputs["directions_path"]))
    rows = {row["name"]: row for row in directions_report["directions"]}
    candidate_spec = config["candidate"]
    names = [
        str(candidate_spec["intervention"]),
        f"matched_random__{candidate_spec['intervention']}",
    ]
    prompt_lengths = {
        row.example_id: (
            int(model.to_tokens(row.initial_prompt).shape[-1]),
            int(model.to_tokens(row.control_prompt).shape[-1]),
            int(model.to_tokens(row.pressured_prompt).shape[-1]),
        )
        for row in examples
    }
    conditions: list[dict[str, Any]] = []
    prediction_rows = [
        {"intervention": "baseline", "strength": 0.0, **row.to_dict()} for row in baseline
    ]
    for index, name in enumerate(names):
        row = rows[name]
        steered = _score(
            model,
            examples,
            str(row["hook_name"]),
            directions[str(row["tensor_key"])],
            float(candidate_spec["strength"]),
            holdout.runtime.max_batch_size,
            holdout.runtime.max_batch_tokens,
            progress,
            name,
            prompt_lengths,
        )
        summary = condition_report(baseline, steered)
        conditions.append(
            {
                "intervention": name,
                "family": row["family"],
                "hook_name": row["hook_name"],
                "strength": candidate_spec["strength"],
                "source_pressure_effect_deltas": _source_deltas(baseline, steered),
                "bootstrap": _bootstrap_interval(
                    baseline,
                    steered,
                    replicates=holdout.protocol.bootstrap_replicates,
                    seed=holdout.runtime.seed + index,
                ),
                **summary,
            }
        )
        prediction_rows.extend(
            {"intervention": name, "strength": candidate_spec["strength"], **item.to_dict()}
            for item in steered
        )
    candidate = conditions[0]
    success = config["success"]
    sources_improved = sum(
        float(value) < 0 for value in candidate["source_pressure_effect_deltas"].values()
    )
    replicated = (
        float(candidate["delta"]["pressure_effect"]) <= -float(success["min_effect_pp"]) / 100
        and max(0.0, -float(candidate["delta"]["initial_accuracy"]))
        <= float(success["max_accuracy_drop_pp"]) / 100
        and sources_improved >= int(success["min_sources_consistent"])
    )
    write_jsonl_records(predictions_path, prediction_rows)
    report = {
        "schema_version": 1,
        "status": (
            "confirmatory_effect_replicated" if replicated else "confirmatory_effect_not_replicated"
        ),
        "split": "replacement_holdout_test",
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "candidate": candidate,
        "matched_random_control": conditions[1],
        "baseline": steering_metrics(baseline, baseline),
        "success_guardrails": success,
        "sources_improved": sources_improved,
        "test_examples": len(examples),
        "test_prompts_scored": True,
        "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
    }
    write_manifest(report_path, report)
    return report


__all__ = [
    "_bootstrap_interval",
    "freeze_phase6e_candidate",
    "load_phase6e_config",
    "run_phase6e_test",
]
