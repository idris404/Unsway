"""Pre-register Phase 6 and build its fresh external evaluation dataset."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from unsway.phase6 import build_phase6_dataset, load_phase6_config, write_protocol_manifest

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
        choices=("protocol", "data"),
        default="data",
        help="Write only the protocol or also acquire and build the fresh dataset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the pre-registration and optionally build the fresh dataset."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_phase6_config(args.config)
    protocol = write_protocol_manifest(config)
    LOGGER.info("Frozen protocol sha256=%s", protocol["protocol_sha256"])
    if args.stage == "protocol":
        return 0
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


if __name__ == "__main__":
    raise SystemExit(main())
