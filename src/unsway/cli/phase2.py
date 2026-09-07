"""Measure the GPT-2 small behavioral sycophancy baseline."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unsway.data.io import load_dataset, write_jsonl_records, write_manifest
from unsway.data.source import sha256_file
from unsway.evaluation.baseline import build_baseline_report
from unsway.evaluation.config import load_phase2_config
from unsway.evaluation.scoring import (
    ExamplePrediction,
    behavior_prediction_sha256,
    dynamic_batches,
    score_batch,
)
from unsway.model import load_transformer
from unsway.runtime import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2.yaml"),
        help="Path to the Phase 2 YAML configuration",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples for a smoke test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs, score all paired prompts, and write the baseline report."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive")
    config = load_phase2_config(args.config)
    actual_dataset_sha = sha256_file(config.dataset.path)
    if actual_dataset_sha != config.dataset.sha256:
        raise ValueError(
            f"Dataset checksum mismatch: {actual_dataset_sha} != {config.dataset.sha256}"
        )
    examples = load_dataset(config.dataset.path)
    if args.limit is not None:
        examples = examples[: args.limit]
    seed_everything(config.seed)
    device = resolve_device(config.model.device)
    LOGGER.info("Loading model=%s on device=%s", config.model.name, device)
    model = load_transformer(config.model)
    batches = list(
        dynamic_batches(
            examples,
            max_batch_size=config.evaluation.max_batch_size,
            max_batch_tokens=config.evaluation.max_batch_tokens,
        )
    )
    LOGGER.info("Evaluating examples=%d batches=%d", len(examples), len(batches))
    started = time.perf_counter()
    predictions: list[ExamplePrediction] = []
    for index, batch in enumerate(batches, start=1):
        predictions.extend(score_batch(model, batch))
        if index == 1 or index % 25 == 0 or index == len(batches):
            LOGGER.info("Progress batches=%d/%d examples=%d", index, len(batches), len(predictions))
    predictions.sort(key=lambda item: item.example_id)
    elapsed_seconds = time.perf_counter() - started
    write_jsonl_records(
        config.output.predictions_path, (prediction.to_dict() for prediction in predictions)
    )
    prediction_sha = sha256_file(config.output.predictions_path)
    metrics = build_baseline_report(predictions)
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "name": config.model.name,
            "device": str(device),
            "dtype": config.model.dtype,
        },
        "dataset": {
            "path": str(config.dataset.path),
            "sha256": actual_dataset_sha,
            "examples": len(examples),
        },
        "evaluation": {
            "seed": config.seed,
            "max_batch_size": config.evaluation.max_batch_size,
            "max_batch_tokens": config.evaluation.max_batch_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "limited": args.limit is not None,
        },
        "predictions_sha256": prediction_sha,
        "behavior_predictions_sha256": behavior_prediction_sha256(predictions),
        "metrics": metrics,
    }
    write_manifest(config.output.report_path, report)
    overall = metrics["overall"]
    LOGGER.info(
        "Phase 2 complete | initial_accuracy=%.4f | eligible=%d | sycophancy_rate=%s | "
        "pressure_effect=%s",
        overall["initial_accuracy"],
        overall["eligible_trials"],
        overall["pressured_target_rate"],
        overall["pressure_effect"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
