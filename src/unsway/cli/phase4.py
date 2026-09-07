"""Sweep anti-sycophancy steering on validation and evaluate once on test."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from unsway.model import load_transformer
from unsway.runtime import seed_everything
from unsway.steering.config import load_phase4_config
from unsway.steering.evaluation import run_frozen_test, run_validation_sweep
from unsway.steering.interventions import build_directions

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 4 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase4.yaml"),
        help="Path to a Phase 4 YAML configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("sweep", "test", "all"),
        default="all",
        help="Validation sweep, frozen test, or both",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the leakage-safe Phase 4 causal intervention protocol."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_phase4_config(args.config)
    seed_everything(config.seed)
    directions = build_directions(config)
    LOGGER.info(
        "Loading model=%s hook=%s interventions=%s",
        config.model.name,
        config.steering.hook_name,
        [item.name for item in directions],
    )
    model = load_transformer(config.model)

    def progress(name: str, strength: float) -> None:
        LOGGER.info("Scoring intervention=%s strength=%g", name, strength)

    if args.stage in {"sweep", "all"}:
        validation = run_validation_sweep(config, model, directions, progress=progress)
        LOGGER.info("Validation selections=%s", validation["selected_strengths"])
    if args.stage in {"test", "all"}:
        test = run_frozen_test(config, model, directions, progress=progress)
        for condition in test["conditions"]:
            LOGGER.info(
                "Test intervention=%s strength=%g pressure_effect=%.4f delta=%.4f "
                "initial_accuracy_delta=%.4f",
                condition["intervention"],
                condition["strength"],
                condition["metrics"]["pressure_effect"],
                condition["delta"]["pressure_effect"],
                condition["delta"]["initial_accuracy"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
