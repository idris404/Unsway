"""Batched multicouche final-token activation extraction for Phase 6."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import save_file

from unsway.data.io import load_dataset, write_manifest
from unsway.data.source import sha256_file
from unsway.evaluation.scoring import behavior_prediction_sha256, dynamic_batches
from unsway.features.corpus import classify_behavior, load_prediction_map
from unsway.phase6.config import Phase6Config
from unsway.phase6.integrity import validate_phase6_inputs

Progress = Callable[[int, int], None]


class MultiCacheModel(Protocol):
    """TransformerLens surface needed for batched multicouche extraction."""

    cfg: Any

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        """Tokenize one prompt or a right-padded prompt batch."""
        ...

    def run_with_cache(
        self, tokens: torch.Tensor, *, names_filter: list[str]
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        """Run a forward pass while caching only selected hooks."""
        ...


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as f:
            temp_path = Path(f.name)
        save_file(tensors, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _baseline_report(config: Phase6Config) -> dict[str, Any]:
    value = json.loads(config.phase6b_output.baseline_report_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 6 baseline report must contain an object")
    return value


def extract_multilayer_activations(
    config: Phase6Config,
    model: MultiCacheModel,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Extract pressured final-token activations from train and validation only."""
    protocol_sha, dataset_sha = validate_phase6_inputs(config)
    baseline = _baseline_report(config)
    if baseline.get("protocol_sha256") != protocol_sha:
        raise ValueError("Phase 6 baseline was produced under a different protocol")
    if baseline.get("status") != "ready_for_frozen_test":
        raise ValueError("Test eligibility guardrail failed; activation extraction is blocked")

    examples = [
        example
        for example in load_dataset(config.dataset.dataset_path)
        if example.split.value in {"train", "validation"}
    ]
    predictions = load_prediction_map(config.phase6b_output.train_validation_predictions_path)
    expected_ids = {example.example_id for example in examples}
    if set(predictions) != expected_ids:
        missing = len(expected_ids - set(predictions))
        unexpected = len(set(predictions) - expected_ids)
        raise ValueError(
            f"Train/validation prediction identity mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    actual_behavior_sha = behavior_prediction_sha256(predictions.values())
    expected_behavior_sha = baseline["train_validation"]["behavior_predictions_sha256"]
    if actual_behavior_sha != expected_behavior_sha:
        raise ValueError("Phase 6 behavior prediction checksum mismatch")

    hooks = list(config.protocol.hook_points)
    split_mapping = {"train": 0, "validation": 1}
    storage_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.runtime.storage_dtype]
    batches = list(
        dynamic_batches(
            examples,
            max_batch_size=config.runtime.max_batch_size,
            max_batch_tokens=config.runtime.max_batch_tokens,
        )
    )
    activation_batches: list[torch.Tensor] = []
    behavior_labels: list[int] = []
    split_codes: list[int] = []
    metadata: list[dict[str, str]] = []
    width = int(model.cfg.d_model)

    for batch_index, batch in enumerate(batches, start=1):
        prompts = [example.pressured_prompt for example in batch]
        tokens = model.to_tokens(prompts)
        lengths = [int(model.to_tokens(prompt).shape[-1]) for prompt in prompts]
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hooks)
        per_hook: list[torch.Tensor] = []
        for hook in hooks:
            if hook not in cache:
                raise KeyError(f"Configured Phase 6 hook was not cached: {hook}")
            values = cache[hook]
            if values.shape[:2] != tokens.shape or values.ndim != 3 or values.shape[-1] != width:
                raise ValueError(
                    f"Hook {hook} has shape {tuple(values.shape)}, expected "
                    f"[{tokens.shape[0]}, {tokens.shape[1]}, {width}]"
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"Hook {hook} contains non-finite activations")
            rows = torch.arange(len(batch), device=values.device)
            positions = torch.tensor(lengths, device=values.device) - 1
            per_hook.append(values[rows, positions].detach().to("cpu", storage_dtype))
        activation_batches.append(torch.stack(per_hook, dim=1).contiguous())
        for example in batch:
            label = classify_behavior(predictions[example.example_id])
            behavior_labels.append(int(label))
            split_codes.append(split_mapping[example.split.value])
            metadata.append(
                {
                    "example_id": example.example_id,
                    "source_dataset": example.source_dataset,
                    "split": example.split.value,
                    "behavior_label": label.name.lower(),
                }
            )
        del cache
        if progress is not None:
            progress(batch_index, len(batches))

    activations = torch.cat(activation_batches, dim=0)
    _atomic_safetensors(
        config.phase6b_output.activations_path,
        {
            "activations": activations,
            "behavior_labels": torch.tensor(behavior_labels, dtype=torch.int8),
            "split_codes": torch.tensor(split_codes, dtype=torch.int8),
        },
    )
    write_manifest(config.phase6b_output.activation_metadata_path, {"examples": metadata})
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "train_validation_multilayer_extracted",
        "protocol_sha256": protocol_sha,
        "dataset_sha256": dataset_sha,
        "behavior_predictions_sha256": actual_behavior_sha,
        "hook_points": hooks,
        "storage_dtype": config.runtime.storage_dtype,
        "shape": list(activations.shape),
        "examples": len(examples),
        "split_counts": dict(Counter(example.split.value for example in examples)),
        "behavior_counts": dict(Counter(item["behavior_label"] for item in metadata)),
        "test_examples_extracted": 0,
        "activations": {
            "path": str(config.phase6b_output.activations_path),
            "sha256": sha256_file(config.phase6b_output.activations_path),
        },
        "metadata": {
            "path": str(config.phase6b_output.activation_metadata_path),
            "sha256": sha256_file(config.phase6b_output.activation_metadata_path),
        },
    }
    write_manifest(config.phase6b_output.extraction_report_path, report)
    return report
