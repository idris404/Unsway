"""Tests for deterministic final-report generation."""

from dataclasses import replace
from pathlib import Path

import pytest

from unsway.reporting.config import Phase5Output, load_phase5_config
from unsway.reporting.figures import build_phase5_artifacts, grouped_bar_svg


def test_grouped_bar_svg_is_accessible_and_escapes_labels() -> None:
    """Generated charts carry accessible titles and escape untrusted labels."""
    svg = grouped_bar_svg(
        "A < B",
        "subtitle",
        ["x&y"],
        [("series", [25.0], "#000000")],
        width=640,
        height=400,
        y_max=100,
        caption="source",
    )

    assert svg.startswith("<svg")
    assert '<title id="title">A &lt; B</title>' in svg
    assert "x&amp;y" in svg
    assert 'role="img"' in svg


def test_grouped_bar_svg_rejects_misaligned_series() -> None:
    """Silent category/series misalignment is rejected."""
    with pytest.raises(ValueError, match="align"):
        grouped_bar_svg(
            "title",
            "subtitle",
            ["a", "b"],
            [("series", [1.0], "#000000")],
            width=640,
            height=400,
            y_max=100,
            caption="source",
        )


def test_phase5_config_file_is_versioned() -> None:
    """The production reporting configuration remains discoverable."""
    assert Path("configs/phase5.yaml").is_file()


def test_phase5_build_generates_complete_artifact_set(tmp_path: Path) -> None:
    """The versioned experiment reports render into five figures and a summary."""
    config = load_phase5_config("configs/phase5.yaml")
    config = replace(
        config,
        output=Phase5Output(
            figure_dir=tmp_path / "figures",
            summary_report=tmp_path / "phase5_summary.json",
        ),
    )

    summary = build_phase5_artifacts(config)

    figures = sorted((tmp_path / "figures").glob("*.svg"))
    assert len(figures) == 5
    assert all(figure.read_text(encoding="utf-8").startswith("<svg") for figure in figures)
    assert summary["conclusion"] == "causal_effect_inconclusive"
    assert (tmp_path / "phase5_summary.json").is_file()
