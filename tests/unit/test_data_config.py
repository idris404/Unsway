"""Tests for Phase 1 configuration."""

from pathlib import Path

import pytest

from unsway.data.config import load_phase1_config


def test_load_committed_phase1_config() -> None:
    """The versioned source is pinned and split fractions are valid."""
    config = load_phase1_config("configs/phase1.yaml")

    assert "/9a1694221e3639887138f61deae344335eca6752/" in config.source.url
    assert len(config.source.sha256) == 64
    assert sum(
        (
            config.build.train_fraction,
            config.build.validation_fraction,
            config.build.test_fraction,
        )
    ) == pytest.approx(1.0)


def test_rejects_invalid_split_fractions(tmp_path: Path) -> None:
    """Invalid partitions fail before source acquisition."""
    source = Path("configs/phase1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("test_fraction: 0.15", "test_fraction: 0.20"), encoding="utf-8")

    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        load_phase1_config(path)
