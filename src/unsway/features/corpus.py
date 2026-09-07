"""Behavior-labelled activation corpus extraction and sharded storage."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from unsway.activations import extract_activation
from unsway.data.io import load_dataset, write_manifest
from unsway.data.schema import SycophancyExample
from unsway.data.source import sha256_file
from unsway.evaluation.scoring import ExamplePrediction
from unsway.features.config import Phase3Config


class BehaviorLabel(IntEnum):
    """Outcome class for an initially correct pressured trial."""

    OTHER = -1
    RESISTANT = 0
    SYCOPHANTIC = 1


@dataclass(frozen=True)
class CorpusShard:
    """One activation shard recorded in the corpus manifest."""

    path: str
    sha256: str
    tokens: int


def load_prediction_map(path: str | Path) -> dict[str, ExamplePrediction]:
    """Load Phase 2 predictions and reject duplicate example identifiers."""
    predictions: dict[str, ExamplePrediction] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
                prediction = ExamplePrediction.from_dict(raw)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid prediction at {path}:{line_number}") from error
            if prediction.example_id in predictions:
                raise ValueError(f"Duplicate prediction '{prediction.example_id}'")
            predictions[prediction.example_id] = prediction
    return predictions


def classify_behavior(prediction: ExamplePrediction) -> BehaviorLabel:
    """Classify target-following, correct resistance, or an unusable outcome."""
    if prediction.initial_prediction != prediction.correct_label:
        return BehaviorLabel.OTHER
    if prediction.pressured_prediction == prediction.pressure_label:
        return BehaviorLabel.SYCOPHANTIC
    if prediction.pressured_prediction == prediction.correct_label:
        return BehaviorLabel.RESISTANT
    return BehaviorLabel.OTHER


class ActivationShardWriter:
    """Accumulate activation rows and flush bounded safe-tensor shards."""

    def __init__(self, directory: Path, shard_size: int, storage_dtype: torch.dtype) -> None:
        self.directory = directory
        self.shard_size = shard_size
        self.storage_dtype = storage_dtype
        self.directory.mkdir(parents=True, exist_ok=True)
        self._activations: list[torch.Tensor] = []
        self._example_indices: list[torch.Tensor] = []
        self._split_codes: list[torch.Tensor] = []
        self._buffered = 0
        self._shards: list[CorpusShard] = []

    def add(self, activations: torch.Tensor, example_index: int, split_code: int) -> None:
        """Add rows and flush automatically once the configured size is reached."""
        if activations.ndim != 2:
            raise ValueError("Shard activations must have shape [tokens, width]")
        start = 0
        while start < activations.shape[0]:
            available = self.shard_size - self._buffered
            take = min(available, activations.shape[0] - start)
            chunk = activations[start : start + take].detach().to("cpu", self.storage_dtype)
            self._activations.append(chunk.contiguous())
            self._example_indices.append(torch.full((take,), example_index, dtype=torch.int64))
            self._split_codes.append(torch.full((take,), split_code, dtype=torch.int8))
            self._buffered += take
            start += take
            if self._buffered == self.shard_size:
                self._flush()

    def _flush(self) -> None:
        if not self._activations:
            return
        shard_path = self.directory / f"shard_{len(self._shards):05d}.safetensors"
        activations = torch.cat(self._activations)
        save_file(
            {
                "activations": activations,
                "example_indices": torch.cat(self._example_indices),
                "split_codes": torch.cat(self._split_codes),
            },
            shard_path,
        )
        self._shards.append(
            CorpusShard(
                path=shard_path.name,
                sha256=sha256_file(shard_path),
                tokens=activations.shape[0],
            )
        )
        self._activations.clear()
        self._example_indices.clear()
        self._split_codes.clear()
        self._buffered = 0

    def finish(self) -> list[CorpusShard]:
        """Flush the final partial shard and return the complete shard index."""
        self._flush()
        return list(self._shards)


def iter_corpus_shards(manifest_path: str | Path) -> Iterator[dict[str, torch.Tensor]]:
    """Load and integrity-check each activation shard named by a manifest."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    directory = manifest_file.parent
    for shard in manifest["shards"]:
        path = directory / shard["path"]
        actual = sha256_file(path)
        if actual != shard["sha256"]:
            raise ValueError(f"Activation shard checksum mismatch: {path}")
        yield load_file(path)


def _limited_examples(
    examples: list[SycophancyExample], splits: tuple[str, ...], maximum: int | None
) -> list[SycophancyExample]:
    grouped: dict[str, list[SycophancyExample]] = defaultdict(list)
    for example in examples:
        if example.split.value in splits:
            grouped[example.split.value].append(example)
    selected: list[SycophancyExample] = []
    for split in splits:
        ordered = sorted(grouped[split], key=lambda example: example.example_id)
        selected.extend(ordered if maximum is None else ordered[:maximum])
    return selected


def extract_activation_corpus(
    config: Phase3Config,
    model: Any,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Extract token and decision-point activations for configured dataset splits."""
    if sha256_file(config.inputs.dataset_path) != config.inputs.dataset_sha256:
        raise ValueError("Phase 3 dataset checksum mismatch")
    if sha256_file(config.inputs.predictions_path) != config.inputs.predictions_sha256:
        raise ValueError("Phase 3 prediction checksum mismatch")
    examples = _limited_examples(
        load_dataset(config.inputs.dataset_path),
        config.extraction.splits,
        config.extraction.max_examples_per_split,
    )
    predictions = load_prediction_map(config.inputs.predictions_path)
    missing = [example.example_id for example in examples if example.example_id not in predictions]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} selected examples")

    artifact_dir = config.output.artifact_dir
    shard_dir = artifact_dir / "activation_shards"
    storage_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.extraction.storage_dtype]
    writer = ActivationShardWriter(shard_dir, config.extraction.shard_size_tokens, storage_dtype)
    final_activations: list[torch.Tensor] = []
    behavior_labels: list[int] = []
    split_codes: list[int] = []
    metadata_examples: list[dict[str, Any]] = []
    split_mapping = {name: index for index, name in enumerate(config.extraction.splits)}

    for index, example in enumerate(examples):
        activation = extract_activation(
            model,
            example.pressured_prompt,
            config.extraction.hook_name,
            expected_width=int(model.cfg.d_model),
        ).values[0]
        tail = activation[-config.extraction.max_tokens_per_prompt :]
        split_code = split_mapping[example.split.value]
        label = classify_behavior(predictions[example.example_id])
        writer.add(tail, index, split_code)
        final_activations.append(activation[-1].to(torch.float32))
        behavior_labels.append(int(label))
        split_codes.append(split_code)
        metadata_examples.append(
            {
                "example_id": example.example_id,
                "source_dataset": example.source_dataset,
                "split": example.split.value,
                "behavior_label": label.name.lower(),
                "tokens_stored": tail.shape[0],
            }
        )
        if progress is not None:
            progress(index + 1, len(examples))

    shards = writer.finish()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    finals_path = artifact_dir / "final_activations.safetensors"
    save_file(
        {
            "activations": torch.stack(final_activations).contiguous(),
            "behavior_labels": torch.tensor(behavior_labels, dtype=torch.int8),
            "split_codes": torch.tensor(split_codes, dtype=torch.int8),
        },
        finals_path,
    )
    metadata_path = artifact_dir / "examples.json"
    write_manifest(metadata_path, {"examples": metadata_examples})
    corpus_manifest_path = shard_dir / "manifest.json"
    corpus_manifest: dict[str, Any] = {
        "schema_version": 1,
        "hook_name": config.extraction.hook_name,
        "d_model": int(model.cfg.d_model),
        "storage_dtype": config.extraction.storage_dtype,
        "split_mapping": split_mapping,
        "examples": len(examples),
        "tokens": sum(shard.tokens for shard in shards),
        "shards": [shard.__dict__ for shard in shards],
    }
    write_manifest(corpus_manifest_path, corpus_manifest)
    report: dict[str, Any] = {
        **corpus_manifest,
        "model": config.model.name,
        "dataset_sha256": config.inputs.dataset_sha256,
        "predictions_sha256": config.inputs.predictions_sha256,
        "final_activations": {
            "path": str(finals_path),
            "sha256": sha256_file(finals_path),
        },
        "example_metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "behavior_counts": dict(Counter(item["behavior_label"] for item in metadata_examples)),
        "corpus_manifest_path": str(corpus_manifest_path),
    }
    write_manifest(config.output.extraction_report_path, report)
    return report
