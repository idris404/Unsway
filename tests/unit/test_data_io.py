"""Tests for dataset persistence."""

import json
from pathlib import Path

import pytest

from unsway.data.io import load_dataset, write_dataset
from unsway.data.schema import Choice, DatasetSplit, SycophancyExample


def example() -> SycophancyExample:
    """Return a minimal valid paired example."""
    return SycophancyExample(
        example_id="example-1",
        source_dataset="test",
        split=DatasetSplit.TEST,
        question="Question?",
        choices=(Choice("A", "true"), Choice("B", "false")),
        correct_label="A",
        pressure_label="B",
        initial_prompt="initial",
        control_prompt="control",
        pressured_prompt="pressured",
        max_prompt_tokens=3,
    )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """Serialized examples preserve their validated schema exactly."""
    path = tmp_path / "dataset.jsonl"
    write_dataset(path, [example()])

    assert load_dataset(path) == [example()]


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    """No example can silently occur in multiple experimental rows."""
    path = tmp_path / "duplicates.jsonl"
    line = json.dumps(example().to_dict())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate example_id"):
        load_dataset(path)
