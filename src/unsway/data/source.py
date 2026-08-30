"""Acquisition and parsing of the public SycophancyEval source dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from unsway.data.schema import Choice

CHOICE_PATTERN = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$")


@dataclass(frozen=True)
class SourceQuestion:
    """Objective multiple-choice fields extracted from a source row."""

    source_dataset: str
    question: str
    choices: tuple[Choice, ...]
    correct_label: str


def sha256_file(path: str | Path) -> str:
    """Compute a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: str | Path, expected_sha256: str) -> Path:
    """Download a source atomically and reject any unexpected content."""
    output_path = Path(destination)
    if output_path.exists():
        actual = sha256_file(output_path)
        if actual == expected_sha256:
            return output_path
        raise ValueError(
            f"Existing source checksum mismatch at {output_path}: {actual} != {expected_sha256}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temp_stream:
            temp_path = Path(temp_stream.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temp_stream.write(chunk)
        actual = sha256_file(temp_path)
        if actual != expected_sha256:
            raise ValueError(f"Downloaded source checksum mismatch: {actual} != {expected_sha256}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return output_path


def parse_choices(raw_answers: str) -> tuple[Choice, ...]:
    """Parse ``(A) answer`` lines from SycophancyEval source records."""
    choices = []
    for line in raw_answers.splitlines():
        match = CHOICE_PATTERN.match(line.strip())
        if match:
            choices.append(Choice(label=match.group(1), text=match.group(2)))
    if len(choices) < 2:
        raise ValueError("Could not parse at least two answer choices")
    return tuple(choices)


def parse_source_row(value: dict[str, Any]) -> SourceQuestion:
    """Extract and validate one objective multiple-choice source record."""
    base = cast(dict[str, Any], value["base"])
    question = str(base["question"]).strip()
    source_dataset = str(base["dataset"])
    correct_label = str(base["correct_letter"]).strip()
    choices = parse_choices(str(base["answers"]))
    labels = {choice.label for choice in choices}
    if not question:
        raise ValueError("Question must not be empty")
    if correct_label not in labels:
        raise ValueError(f"Correct label '{correct_label}' is absent from parsed choices")
    return SourceQuestion(source_dataset, question, choices, correct_label)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects with line-numbered parsing errors."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield cast(dict[str, Any], value)
