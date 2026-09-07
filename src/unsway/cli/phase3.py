"""Extract activations, train a Top-K SAE, and identify behavioral features."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from unsway.features.analysis import analyze_features
from unsway.features.config import load_phase3_config
from unsway.features.corpus import extract_activation_corpus
from unsway.features.sae import train_sae
from unsway.model import load_transformer
from unsway.runtime import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase3.yaml"),
        help="Path to a Phase 3 YAML configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("extract", "train", "analyze", "all"),
        default="all",
        help="Pipeline stage to execute",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one stage or the complete Phase 3 pipeline."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_phase3_config(args.config)
    seed_everything(config.seed)

    if args.stage in {"extract", "all"}:
        device = resolve_device(config.model.device)
        LOGGER.info(
            "Extracting hook=%s model=%s device=%s",
            config.extraction.hook_name,
            config.model.name,
            device,
        )
        model = load_transformer(config.model)

        def extraction_progress(done: int, total: int) -> None:
            if done == 1 or done % 25 == 0 or done == total:
                LOGGER.info("Extraction progress examples=%d/%d", done, total)

        extraction = extract_activation_corpus(config, model, progress=extraction_progress)
        LOGGER.info(
            "Extraction complete examples=%d tokens=%d behavior_counts=%s",
            extraction["examples"],
            extraction["tokens"],
            extraction["behavior_counts"],
        )

    if args.stage in {"train", "all"}:
        LOGGER.info(
            "Training Top-K SAE expansion=%d k=%d epochs=%d",
            config.sae.expansion_factor,
            config.sae.k,
            config.sae.epochs,
        )

        def training_progress(epoch: int, total: int, metrics: dict[str, object]) -> None:
            validation = metrics["validation"]
            LOGGER.info(
                "Epoch=%d/%d train_mse=%.6f validation=%s",
                epoch,
                total,
                metrics["train_mse"],
                validation,
            )

        training = train_sae(config, progress=training_progress)
        LOGGER.info("SAE training complete artifact=%s", training["artifact"])

    if args.stage in {"analyze", "all"}:
        LOGGER.info("Ranking SAE features on train labels")
        feature_report = analyze_features(config)
        top = feature_report["top_features"][0]
        LOGGER.info(
            "Feature analysis complete top_feature=%d orientation=%s "
            "train_discrimination_auroc=%.4f validation_oriented_auroc=%.4f",
            top["train"]["feature_index"],
            top["train"]["orientation"],
            top["train"]["discrimination_auroc"],
            top["validation"]["oriented_auroc"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
