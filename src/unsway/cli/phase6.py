"""Run pre-registered Phase 6 data, baseline, and extraction stages."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from unsway.model import load_transformer
from unsway.phase6 import (
    build_phase6_dataset,
    extract_multilayer_activations,
    load_phase6_config,
    run_phase6_baseline,
    write_protocol_manifest,
)
from unsway.runtime import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 6 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase6.yaml"),
        help="Path to the frozen Phase 6 YAML configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("protocol", "data", "baseline", "extract", "phase6b"),
        default="data",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--eligibility-config",
        type=Path,
        help=(
            "Frozen replacement-holdout config that may unlock extraction when the "
            "training protocol's retired test narrowly missed its guardrail"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the pre-registration and optionally build the fresh dataset."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_phase6_config(args.config)
    eligibility_config = (
        None if args.eligibility_config is None else load_phase6_config(args.eligibility_config)
    )
    protocol = write_protocol_manifest(config)
    LOGGER.info("Frozen protocol sha256=%s", protocol["protocol_sha256"])
    if args.stage == "protocol":
        return 0
    if args.stage == "data":
        if not config.dataset.phase1_dataset_path.is_file():
            raise FileNotFoundError(
                f"Phase 1 dataset is required for overlap checks: "
                f"{config.dataset.phase1_dataset_path}. Run unsway-phase1 first."
            )
        LOGGER.info("Loading tokenizer=%s", config.dataset.tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_name)
        tokenizer.model_max_length = 1_000_000

        def count_tokens(text: str) -> int:
            encoded: Any = tokenizer.encode(text, add_special_tokens=False)
            return len(encoded)

        manifest = build_phase6_dataset(config, count_tokens)
        report = manifest["report"]
        LOGGER.info(
            "Fresh dataset built | examples=%d | sources=%s | splits=%s",
            report["examples"],
            report["source_counts"],
            report["split_counts"],
        )
        LOGGER.info("dataset_sha256=%s", manifest["dataset_sha256"])
        return 0

    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.model.device)
    LOGGER.info("Loading model=%s device=%s", config.runtime.model.name, device)
    model = load_transformer(config.runtime.model)
    if args.stage in {"baseline", "phase6b"}:

        def baseline_progress(partition: str, done: int, total: int) -> None:
            if done == 1 or done % 25 == 0 or done == total:
                LOGGER.info("Baseline %s batches=%d/%d", partition, done, total)

        baseline = run_phase6_baseline(config, model, progress=baseline_progress)
        test_summary = baseline["test_initial_only"]["metrics"]["overall"]
        LOGGER.info(
            "Baseline complete status=%s test_initial_correct=%d/%d",
            baseline["status"],
            test_summary["initial_correct_trials"],
            test_summary["trials"],
        )
        if baseline["status"] != "ready_for_frozen_test":
            raise RuntimeError("Phase 6 minimum test-eligibility guardrail was not met")
    if args.stage in {"extract", "phase6b"}:

        def extraction_progress(done: int, total: int) -> None:
            if done == 1 or done % 25 == 0 or done == total:
                LOGGER.info("Multilayer extraction batches=%d/%d", done, total)

        extraction = extract_multilayer_activations(
            config,
            model,
            eligibility_config=eligibility_config,
            progress=extraction_progress,
        )
        LOGGER.info(
            "Extraction complete examples=%d shape=%s behavior_counts=%s",
            extraction["examples"],
            extraction["shape"],
            extraction["behavior_counts"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
