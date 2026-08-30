"""Deterministic JSONL persistence for experimental datasets."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from unsway.data.schema import SycophancyExample


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_dataset(path: str | Path, examples: Iterable[SycophancyExample]) -> None:
    """Atomically write examples as stable, UTF-8 JSONL."""
    lines = [
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for example in examples
    ]
    _atomic_text_write(Path(path), "\n".join(lines) + ("\n" if lines else ""))


def load_dataset(path: str | Path) -> list[SycophancyExample]:
    """Load a JSONL dataset and reject duplicate identifiers."""
    examples: list[SycophancyExample] = []
    identifiers: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                example = SycophancyExample.from_dict(cast(dict[str, Any], value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid dataset record at {path}:{line_number}") from error
            if example.example_id in identifiers:
                raise ValueError(f"Duplicate example_id '{example.example_id}'")
            identifiers.add(example.example_id)
            examples.append(example)
    return examples


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Atomically write a human-readable reproducibility manifest."""
    _atomic_text_write(
        Path(path), json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
