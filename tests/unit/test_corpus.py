"""Tests for behavior labels and safe activation shards."""

from pathlib import Path

import torch

from unsway.data.io import write_manifest
from unsway.evaluation.scoring import ExamplePrediction
from unsway.features.corpus import (
    ActivationShardWriter,
    BehaviorLabel,
    classify_behavior,
    iter_corpus_shards,
)


def prediction(initial: str, pressured: str) -> ExamplePrediction:
    """Create a minimal Phase 2 prediction."""
    return ExamplePrediction(
        example_id="id",
        source_dataset="source",
        split="train",
        correct_label="A",
        pressure_label="B",
        initial_scores={"A": 0.0, "B": -1.0},
        control_scores={"A": 0.0, "B": -1.0},
        pressured_scores={"A": -1.0, "B": 0.0},
        initial_prediction=initial,
        control_prediction="A",
        pressured_prediction=pressured,
    )


def test_behavior_classification_requires_initial_correctness() -> None:
    """Only targeted flips and correct resistance receive analysis labels."""
    assert classify_behavior(prediction("A", "B")) == BehaviorLabel.SYCOPHANTIC
    assert classify_behavior(prediction("A", "A")) == BehaviorLabel.RESISTANT
    assert classify_behavior(prediction("B", "B")) == BehaviorLabel.OTHER
    assert classify_behavior(prediction("A", "C")) == BehaviorLabel.OTHER


def test_shards_are_bounded_and_integrity_checked(tmp_path: Path) -> None:
    """Rows spanning boundaries are split without loss and read from the manifest."""
    writer = ActivationShardWriter(tmp_path, shard_size=3, storage_dtype=torch.float16)
    writer.add(torch.arange(20).reshape(5, 4), example_index=7, split_code=0)
    shards = writer.finish()
    assert [shard.tokens for shard in shards] == [3, 2]

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "split_mapping": {"train": 0},
            "shards": [shard.__dict__ for shard in shards],
        },
    )
    loaded = list(iter_corpus_shards(manifest_path))

    assert torch.cat([item["activations"] for item in loaded]).shape == (5, 4)
    assert torch.equal(torch.cat([item["example_indices"] for item in loaded]), torch.full((5,), 7))
