"""Generate the final Unsway research figures and summary manifest."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from unsway.reporting import build_phase5_artifacts
from unsway.reporting.config import load_phase5_config

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 5 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase5.yaml"),
        help="Path to a Phase 5 YAML configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate deterministic final-report artifacts from versioned manifests."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    summary = build_phase5_artifacts(load_phase5_config(args.config))
    LOGGER.info("Generated %d figures", len(summary["figures"]))
    LOGGER.info("Conclusion=%s", summary["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
