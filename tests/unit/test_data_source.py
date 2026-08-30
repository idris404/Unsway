"""Tests for source parsing and integrity helpers."""

import hashlib
from pathlib import Path

import pytest

from unsway.data.source import download_verified, parse_choices, parse_source_row


def test_parse_choices_preserves_labels_and_text() -> None:
    """Multiple-choice text is converted without losing answer content."""
    choices = parse_choices("\n(A) Alpha\n(B) Beta with spaces\n(C) Gamma")

    assert [(choice.label, choice.text) for choice in choices] == [
        ("A", "Alpha"),
        ("B", "Beta with spaces"),
        ("C", "Gamma"),
    ]


def test_parse_source_row_rejects_unknown_correct_label() -> None:
    """Corrupt ground-truth labels cannot enter the experiment."""
    row = {
        "base": {
            "dataset": "test",
            "question": "Question?",
            "correct_letter": "C",
            "answers": "(A) one\n(B) two",
        }
    }

    with pytest.raises(ValueError, match="absent"):
        parse_source_row(row)


def test_download_verified_reuses_valid_local_file(tmp_path: Path) -> None:
    """A cached source is accepted only when its digest matches."""
    path = tmp_path / "source.jsonl"
    content = b'{"valid":true}\n'
    path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()

    assert download_verified("https://invalid.example", path, checksum) == path

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_verified("https://invalid.example", path, "0" * 64)
