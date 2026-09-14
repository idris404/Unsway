"""Validate versioned YAML, JSON, and Jupyter notebook artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    """Parse structured artifacts and verify the basic notebook schema."""
    checked = 0
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"YAML root must be a mapping: {path.relative_to(ROOT)}")
        checked += 1

    for path in sorted((ROOT / "reports").rglob("*.json")):
        value = _load_json(path)
        if not isinstance(value, (dict, list)):
            raise ValueError(f"JSON root must be an object or array: {path.relative_to(ROOT)}")
        checked += 1

    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        notebook = _load_json(path)
        if not isinstance(notebook, dict):
            raise ValueError(f"Notebook root must be an object: {path.relative_to(ROOT)}")
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            raise ValueError(f"Invalid notebook schema: {path.relative_to(ROOT)}")
        for index, cell in enumerate(notebook["cells"]):
            if not isinstance(cell, dict) or cell.get("cell_type") not in {
                "code",
                "markdown",
                "raw",
            }:
                raise ValueError(f"Invalid notebook cell: {path.relative_to(ROOT)}:{index}")
            if not isinstance(cell.get("source"), list):
                raise ValueError(f"Notebook cell source must be a list: {path.relative_to(ROOT)}")
        checked += 1

    print(f"Validated {checked} structured repository artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
