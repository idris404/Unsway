"""Phase 0 end-to-end activation extraction smoke test."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from unsway.activations import extract_activation
from unsway.config import load_config
from unsway.model import load_transformer
from unsway.runtime import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 0 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase0.yaml"),
        help="Path to the Phase 0 YAML configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load GPT-2, extract one activation, and report the validated shape."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(config.model.device)
    LOGGER.info("Loading model=%s on device=%s", config.model.name, device)
    model = load_transformer(config.model)
    activation = extract_activation(
        model,
        config.prompt,
        config.activation.hook_name,
        expected_width=config.activation.expected_width,
    )
    LOGGER.info(
        "Phase 0 validation passed | hook=%s | shape=%s | dtype=%s | finite=true",
        activation.hook_name,
        activation.shape,
        activation.values.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
