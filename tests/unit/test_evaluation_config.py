"""Tests for Phase 2 configuration."""

from pathlib import Path

import pytest

from unsway.evaluation.config import load_phase2_config


def test_load_committed_phase2_config() -> None:
    """The baseline is tied to the exact Phase 1 dataset."""
    config = load_phase2_config("configs/phase2.yaml")

    assert config.dataset.sha256 == (
        "d8abd363943e1983bfd4df2af8e85bc0c00c24f8e45a8f11bbd24432cbff4c9f"
    )
    assert config.evaluation.max_batch_tokens >= 1024


def test_rejects_invalid_batch_limit(tmp_path: Path) -> None:
    """Invalid memory limits fail before model loading."""
    source = Path("configs/phase2.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("max_batch_size: 16", "max_batch_size: 0"), encoding="utf-8")

    with pytest.raises(ValueError, match="batch size must be positive"):
        load_phase2_config(path)
