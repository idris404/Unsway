"""Cross-stage integrity checks for the pre-registered Phase 6 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from unsway.data.source import sha256_file
from unsway.phase6.config import Phase6Config, protocol_sha256


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must contain an object: {path}")
    return cast(dict[str, Any], value)


def validate_phase6_inputs(config: Phase6Config) -> tuple[str, str]:
    """Validate protocol and fresh-dataset identity before model execution."""
    expected_protocol = protocol_sha256(config)
    protocol_manifest = _manifest(config.protocol.manifest_path)
    if protocol_manifest.get("protocol_sha256") != expected_protocol:
        raise ValueError("Current Phase 6 configuration does not match the frozen protocol")

    dataset_manifest = _manifest(config.dataset.manifest_path)
    if dataset_manifest.get("protocol_sha256") != expected_protocol:
        raise ValueError("Phase 6 dataset was built under a different protocol")
    expected_dataset = str(dataset_manifest.get("dataset_sha256", ""))
    actual_dataset = sha256_file(config.dataset.dataset_path)
    if actual_dataset != expected_dataset:
        raise ValueError(
            f"Phase 6 dataset checksum mismatch: {actual_dataset} != {expected_dataset}"
        )
    return expected_protocol, actual_dataset
