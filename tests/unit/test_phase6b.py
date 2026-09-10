"""Leakage and shape tests for the Phase 6B model run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from unsway.data.io import write_dataset, write_manifest
from unsway.data.schema import Choice, DatasetSplit, SycophancyExample
from unsway.data.source import sha256_file
from unsway.phase6.activations import extract_multilayer_activations
from unsway.phase6.baseline import run_phase6_baseline
from unsway.phase6.config import (
    Phase6BOutput,
    Phase6Config,
    load_phase6_config,
    protocol_sha256,
)
from unsway.phase6.dataset import write_protocol_manifest


class FakeTokenizer:
    """Map answer labels to stable token identifiers."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [ord(text) - ord("A") + 1]


class FakePhase6Model:
    """Always choose A and expose deterministic activation caches."""

    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(d_model=4)
        self.seen_texts: list[str] = []

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        texts = [text] if isinstance(text, str) else text
        self.seen_texts.extend(texts)
        width = max(len(item) for item in texts)
        return torch.zeros((len(texts), width), dtype=torch.long)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, positions = tokens.shape
        logits = torch.zeros((batch, positions, 8), dtype=torch.float32)
        logits[:, :, 1] = 5.0
        return logits

    def run_with_cache(
        self, tokens: torch.Tensor, *, names_filter: list[str]
    ) -> tuple[None, dict[str, torch.Tensor]]:
        batch, positions = tokens.shape
        cache = {
            hook: torch.full((batch, positions, 4), float(index), dtype=torch.float32)
            for index, hook in enumerate(names_filter)
        }
        return None, cache


def _example(identifier: str, split: DatasetSplit) -> SycophancyExample:
    return SycophancyExample(
        example_id=identifier,
        source_dataset="source",
        split=split,
        question=f"question-{identifier}",
        choices=(Choice("A", "correct"), Choice("B", "incorrect")),
        correct_label="A",
        pressure_label="B",
        initial_prompt=f"initial-{identifier}",
        control_prompt=f"control-{identifier}",
        pressured_prompt=f"pressured-{identifier}",
        max_prompt_tokens=32,
    )


def _config(tmp_path: Path) -> tuple[Phase6Config, list[SycophancyExample]]:
    examples = [
        _example("train-1", DatasetSplit.TRAIN),
        _example("train-2", DatasetSplit.TRAIN),
        _example("validation-1", DatasetSplit.VALIDATION),
        _example("test-1", DatasetSplit.TEST),
        _example("test-2", DatasetSplit.TEST),
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    write_dataset(dataset_path, examples)
    base = load_phase6_config("configs/phase6.yaml")
    output = Phase6BOutput(
        train_validation_predictions_path=tmp_path / "train_validation.jsonl",
        test_initial_predictions_path=tmp_path / "test_initial.jsonl",
        baseline_report_path=tmp_path / "baseline.json",
        activations_path=tmp_path / "activations.safetensors",
        activation_metadata_path=tmp_path / "metadata.json",
        extraction_report_path=tmp_path / "extraction.json",
    )
    config = replace(
        base,
        dataset=replace(
            base.dataset,
            dataset_path=dataset_path,
            manifest_path=tmp_path / "dataset_manifest.json",
        ),
        protocol=replace(
            base.protocol,
            min_test_eligible=2,
            manifest_path=tmp_path / "protocol.json",
        ),
        runtime=replace(base.runtime, max_batch_size=2, max_batch_tokens=128),
        phase6b_output=output,
    )
    write_protocol_manifest(config)
    write_manifest(
        config.dataset.manifest_path,
        {
            "protocol_sha256": protocol_sha256(config),
            "dataset_sha256": sha256_file(dataset_path),
        },
    )
    return config, examples


def test_baseline_never_scores_test_pressure_or_control(tmp_path: Path) -> None:
    config, examples = _config(tmp_path)
    model = FakePhase6Model()

    report = run_phase6_baseline(config, model)

    forbidden = {
        prompt
        for example in examples
        if example.split == DatasetSplit.TEST
        for prompt in (example.control_prompt, example.pressured_prompt)
    }
    assert forbidden.isdisjoint(model.seen_texts)
    test_rows = [
        json.loads(line)
        for line in config.phase6b_output.test_initial_predictions_path.read_text().splitlines()
    ]
    assert len(test_rows) == 2
    assert set(test_rows[0]) == {
        "correct_label",
        "example_id",
        "initial_prediction",
        "initial_scores",
        "source_dataset",
        "split",
    }
    assert report["status"] == "ready_for_frozen_test"
    assert report["test_initial_only"]["pressure_scored"] is False
    assert report["test_initial_only"]["control_scored"] is False


def test_baseline_refuses_to_overwrite_frozen_test_outputs(tmp_path: Path) -> None:
    config, _examples = _config(tmp_path)
    model = FakePhase6Model()
    run_phase6_baseline(config, model)
    model.seen_texts.clear()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_phase6_baseline(config, model)

    assert model.seen_texts == []


def test_multilayer_extraction_contains_no_test_examples(tmp_path: Path) -> None:
    config, _examples = _config(tmp_path)
    model = FakePhase6Model()
    run_phase6_baseline(config, model)
    model.seen_texts.clear()

    report = extract_multilayer_activations(config, model)

    tensors = load_file(config.phase6b_output.activations_path)
    assert tuple(tensors["activations"].shape) == (3, 12, 4)
    assert report["test_examples_extracted"] == 0
    assert report["split_counts"] == {"train": 2, "validation": 1}
    assert all("test" not in prompt for prompt in model.seen_texts)


def test_replacement_holdout_can_unlock_training_only_extraction(tmp_path: Path) -> None:
    training_config, _examples = _config(tmp_path / "training")
    training_config = replace(
        training_config,
        protocol=replace(
            training_config.protocol,
            min_test_eligible=3,
            manifest_path=tmp_path / "training" / "protocol-insufficient.json",
        ),
    )
    write_protocol_manifest(training_config)
    write_manifest(
        training_config.dataset.manifest_path,
        {
            "protocol_sha256": protocol_sha256(training_config),
            "dataset_sha256": sha256_file(training_config.dataset.dataset_path),
        },
    )
    model = FakePhase6Model()
    training_report = run_phase6_baseline(training_config, model)
    assert training_report["status"] == "insufficient_test_eligibility"

    eligibility_config, _gate_examples = _config(tmp_path / "eligibility")
    eligibility_config = replace(
        eligibility_config,
        protocol=replace(
            eligibility_config.protocol,
            experiment_id="replacement-holdout",
            manifest_path=tmp_path / "eligibility" / "protocol-replacement.json",
        ),
    )
    write_protocol_manifest(eligibility_config)
    write_manifest(
        eligibility_config.dataset.manifest_path,
        {
            "protocol_sha256": protocol_sha256(eligibility_config),
            "dataset_sha256": sha256_file(eligibility_config.dataset.dataset_path),
        },
    )
    gate_report = run_phase6_baseline(eligibility_config, FakePhase6Model())
    assert gate_report["status"] == "ready_for_frozen_test"

    model.seen_texts.clear()
    report = extract_multilayer_activations(
        training_config,
        model,
        eligibility_config=eligibility_config,
    )

    assert report["eligibility_gate"]["mode"] == "external_replacement_holdout"
    assert report["eligibility_gate"]["initial_correct_trials"] == 2
    assert report["test_examples_extracted"] == 0
    assert all("test" not in prompt for prompt in model.seen_texts)


def test_replacement_gate_rejects_tampered_predictions(tmp_path: Path) -> None:
    training_config, _examples = _config(tmp_path / "training")
    training_config = replace(
        training_config,
        protocol=replace(
            training_config.protocol,
            min_test_eligible=3,
            manifest_path=tmp_path / "training" / "protocol-insufficient.json",
        ),
    )
    write_protocol_manifest(training_config)
    write_manifest(
        training_config.dataset.manifest_path,
        {
            "protocol_sha256": protocol_sha256(training_config),
            "dataset_sha256": sha256_file(training_config.dataset.dataset_path),
        },
    )
    run_phase6_baseline(training_config, FakePhase6Model())

    eligibility_config, _gate_examples = _config(tmp_path / "eligibility")
    eligibility_config = replace(
        eligibility_config,
        protocol=replace(
            eligibility_config.protocol,
            experiment_id="replacement-holdout",
            manifest_path=tmp_path / "eligibility" / "protocol-replacement.json",
        ),
    )
    write_protocol_manifest(eligibility_config)
    write_manifest(
        eligibility_config.dataset.manifest_path,
        {
            "protocol_sha256": protocol_sha256(eligibility_config),
            "dataset_sha256": sha256_file(eligibility_config.dataset.dataset_path),
        },
    )
    run_phase6_baseline(eligibility_config, FakePhase6Model())
    eligibility_config.phase6b_output.test_initial_predictions_path.write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_multilayer_activations(
            training_config,
            FakePhase6Model(),
            eligibility_config=eligibility_config,
        )
