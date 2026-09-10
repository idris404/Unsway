"""Tests for fresh external dataset construction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from unsway.data.io import load_dataset, write_dataset
from unsway.data.schema import Choice, DatasetSplit, SycophancyExample
from unsway.data.source import sha256_file
from unsway.phase6.config import ArtifactConfig, ArtifactFormat, load_phase6_config
from unsway.phase6.dataset import build_phase6_dataset, normalize_question


def _row(question: str) -> dict[str, object]:
    return {
        "id": question,
        "question": {
            "stem": question,
            "choices": [
                {"label": "1", "text": "correct"},
                {"label": "2", "text": "incorrect"},
            ],
        },
        "answerKey": "1",
    }


def _write_rows(path: Path, questions: list[str]) -> None:
    path.write_text(
        "\n".join(json.dumps(_row(question)) for question in questions) + "\n",
        encoding="utf-8",
    )


def _reference_example(question: str) -> SycophancyExample:
    choices = (Choice("A", "correct"), Choice("B", "incorrect"))
    return SycophancyExample(
        example_id="reference",
        source_dataset="phase1",
        split=DatasetSplit.TRAIN,
        question=question,
        choices=choices,
        correct_label="A",
        pressure_label="B",
        initial_prompt="initial",
        control_prompt="control",
        pressured_prompt="pressured",
        max_prompt_tokens=1,
    )


def test_normalize_question_ignores_case_spacing_and_punctuation() -> None:
    assert normalize_question("  What IS this? ") == normalize_question("what is this")


def test_phase6_dataset_is_balanced_and_excludes_phase1_overlap(tmp_path: Path) -> None:
    source_one = tmp_path / "one.jsonl"
    source_two = tmp_path / "two.jsonl"
    overlap = "Previously used question"
    _write_rows(source_one, [overlap, *(f"one question {index}" for index in range(4))])
    _write_rows(source_two, [f"two question {index}" for index in range(5)])
    reference = tmp_path / "phase1.jsonl"
    write_dataset(reference, [_reference_example(overlap)])

    base = load_phase6_config("configs/phase6.yaml")
    artifacts = tuple(
        ArtifactConfig(
            source=name,
            url="https://invalid.example/not-used",
            sha256=sha256_file(path),
            path=path,
            format=ArtifactFormat.JSONL,
        )
        for name, path in (("one", source_one), ("two", source_two))
    )
    config = replace(
        base,
        artifacts=artifacts,
        dataset=replace(
            base.dataset,
            max_examples_per_source=4,
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
            phase1_dataset_path=reference,
            dataset_path=tmp_path / "phase6.jsonl",
            manifest_path=tmp_path / "dataset_manifest.json",
        ),
        protocol=replace(
            base.protocol,
            min_sources_consistent=2,
            manifest_path=tmp_path / "protocol.json",
        ),
    )

    manifest = build_phase6_dataset(config, lambda text: len(text.split()))

    assert manifest["report"]["examples"] == 8
    assert manifest["report"]["source_counts"] == {"one": 4, "two": 4}
    assert manifest["report"]["split_counts"] == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert manifest["report"]["source_split_counts"] == {
        "one": {"train": 2, "validation": 1, "test": 1},
        "two": {"train": 2, "validation": 1, "test": 1},
    }
    assert manifest["report"]["external_excluded_counts"]["one:phase1_overlap"] == 1
    assert config.dataset.dataset_path.is_file()


def test_replacement_holdout_excludes_every_prior_phase_question(tmp_path: Path) -> None:
    source_one = tmp_path / "one.jsonl"
    source_two = tmp_path / "two.jsonl"
    _write_rows(source_one, ["old one", "new one", "spare one"])
    _write_rows(source_two, ["old two", "new two", "spare two"])
    phase1 = tmp_path / "phase1.jsonl"
    prior = tmp_path / "prior.jsonl"
    write_dataset(phase1, [_reference_example("unrelated phase1 question")])
    write_dataset(
        prior,
        [
            replace(_reference_example("old one"), example_id="prior-one"),
            replace(_reference_example("old two"), example_id="prior-two"),
        ],
    )

    base = load_phase6_config("configs/phase6.yaml")
    artifacts = tuple(
        ArtifactConfig(
            source=name,
            url="https://invalid.example/not-used",
            sha256=sha256_file(path),
            path=path,
            format=ArtifactFormat.JSONL,
        )
        for name, path in (("one", source_one), ("two", source_two))
    )
    config = replace(
        base,
        artifacts=artifacts,
        dataset=replace(
            base.dataset,
            max_examples_per_source=1,
            train_fraction=0,
            validation_fraction=0,
            test_fraction=1,
            phase1_dataset_path=phase1,
            dataset_path=tmp_path / "replacement.jsonl",
            manifest_path=tmp_path / "replacement_manifest.json",
            expected_examples=2,
            exclusion_dataset_paths=(prior,),
            exclusion_dataset_sha256s=(sha256_file(prior),),
        ),
        protocol=replace(
            base.protocol,
            min_sources_consistent=2,
            manifest_path=tmp_path / "replacement_protocol.json",
        ),
    )

    manifest = build_phase6_dataset(config, lambda text: len(text.split()))
    examples = load_dataset(config.dataset.dataset_path)

    assert manifest["report"]["split_counts"] == {"test": 2}
    assert {example.question for example in examples}.isdisjoint({"old one", "old two"})
    assert manifest["report"]["external_excluded_counts"] == {
        "one:prior_phase_overlap": 1,
        "one:source_cap": 1,
        "two:prior_phase_overlap": 1,
        "two:source_cap": 1,
    }
