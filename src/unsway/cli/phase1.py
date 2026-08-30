"""Build the reproducible paired sycophancy dataset for Phase 1."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from unsway.data.builder import build_dataset
from unsway.data.config import load_phase1_config
from unsway.data.io import write_dataset, write_manifest
from unsway.data.source import download_verified, iter_jsonl, sha256_file

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1.yaml"),
        help="Path to the Phase 1 YAML configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Acquire, validate, transform, split, and persist the Phase 1 dataset."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_phase1_config(args.config)
    LOGGER.info("Acquiring pinned source=%s", config.source.name)
    source_path = download_verified(config.source.url, config.source.path, config.source.sha256)
    LOGGER.info("Loading tokenizer=%s", config.tokenizer.name)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    tokenizer.model_max_length = 1_000_000

    def count_tokens(text: str) -> int:
        encoded: Any = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)

    examples, report = build_dataset(
        iter_jsonl(source_path),
        config.build,
        count_tokens,
        max_prompt_tokens=config.tokenizer.max_prompt_tokens,
    )
    write_dataset(config.output.dataset_path, examples)
    dataset_sha256 = sha256_file(config.output.dataset_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "name": config.source.name,
            "url": config.source.url,
            "sha256": config.source.sha256,
        },
        "tokenizer": {
            "name": config.tokenizer.name,
            "max_prompt_tokens": config.tokenizer.max_prompt_tokens,
        },
        "build": {
            "seed": config.build.seed,
            "allowed_datasets": list(config.build.allowed_datasets),
            "split_fractions": {
                "train": config.build.train_fraction,
                "validation": config.build.validation_fraction,
                "test": config.build.test_fraction,
            },
        },
        "report": {
            "source_rows": report.source_rows,
            "examples": report.examples,
            "split_counts": report.split_counts,
            "source_counts": report.source_counts,
            "excluded_counts": report.excluded_counts,
        },
        "dataset_sha256": dataset_sha256,
    }
    write_manifest(config.output.manifest_path, manifest)
    LOGGER.info(
        "Phase 1 dataset built | examples=%d | splits=%s | sha256=%s",
        report.examples,
        report.split_counts,
        dataset_sha256,
    )
    LOGGER.info("Exclusions=%s", report.excluded_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
