"""Build a fresh, checksum-pinned Phase 6 multiple-choice dataset."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as parquet

from unsway.data.builder import build_dataset
from unsway.data.config import BuildConfig
from unsway.data.io import load_dataset, write_dataset, write_manifest
from unsway.data.schema import Choice
from unsway.data.source import SourceQuestion, download_verified, iter_jsonl, sha256_file
from unsway.phase6.config import (
    ArtifactConfig,
    ArtifactFormat,
    Phase6Config,
    protocol_payload,
    protocol_sha256,
)

TokenCounter = Callable[[str], int]
NORMALIZED_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def normalize_question(text: str) -> str:
    """Normalize question text for deterministic cross-dataset deduplication."""
    return " ".join(NORMALIZED_TOKEN.findall(text.casefold()))


def _parse_ai2_row(row: dict[str, Any], source: str) -> SourceQuestion:
    raw_question = row["question"]
    if isinstance(raw_question, dict):
        question = str(raw_question["stem"]).strip()
        raw_choices = raw_question["choices"]
    else:
        question = str(raw_question).strip()
        raw_choices = row["choices"]

    pairs: list[tuple[str, str]] = []
    if isinstance(raw_choices, list):
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                raise ValueError("Choice rows must be mappings")
            pairs.append((str(raw_choice["label"]), str(raw_choice["text"]).strip()))
    elif isinstance(raw_choices, dict):
        labels = raw_choices.get("label")
        texts = raw_choices.get("text")
        if not isinstance(labels, list) or not isinstance(texts, list) or len(labels) != len(texts):
            raise ValueError("Columnar choices must contain aligned label and text lists")
        pairs.extend(
            (str(label), str(text).strip()) for label, text in zip(labels, texts, strict=True)
        )
    else:
        raise ValueError("Unsupported choices representation")

    if not question or not 2 <= len(pairs) <= 8 or any(not text for _, text in pairs):
        raise ValueError("Question and two to eight non-empty choices are required")
    original_labels = [label for label, _ in pairs]
    if len(original_labels) != len(set(original_labels)):
        raise ValueError("Choice labels must be unique")
    answer_key = str(row["answerKey"]).strip()
    if answer_key in original_labels:
        correct_index = original_labels.index(answer_key)
    elif answer_key.isdigit() and 1 <= int(answer_key) <= len(pairs):
        correct_index = int(answer_key) - 1
    else:
        raise ValueError("Answer key does not identify a choice")

    choices = tuple(
        Choice(label=chr(ord("A") + index), text=text) for index, (_label, text) in enumerate(pairs)
    )
    return SourceQuestion(
        source_dataset=source,
        question=question,
        choices=choices,
        correct_label=choices[correct_index].label,
    )


def _iter_zip_rows(path: Path, members: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        available = set(archive.namelist())
        for member in members:
            if member not in available:
                raise ValueError(f"Configured ZIP member is missing: {member}")
            with archive.open(member) as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        value = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"Invalid JSON in {path}:{member}:{line_number}"
                        ) from error
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected object in {path}:{member}:{line_number}")
                    yield cast(dict[str, Any], value)


def _iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    table = parquet.read_table(path, columns=["id", "question", "choices", "answerKey"])
    for row in table.to_pylist():
        if not isinstance(row, dict):
            raise ValueError(f"Expected Parquet object row in {path}")
        yield cast(dict[str, Any], row)


def _artifact_rows(artifact: ArtifactConfig) -> Iterable[dict[str, Any]]:
    if artifact.format is ArtifactFormat.JSONL:
        return iter_jsonl(artifact.path)
    if artifact.format is ArtifactFormat.ZIP_JSONL:
        return _iter_zip_rows(artifact.path, artifact.members)
    if artifact.format is ArtifactFormat.PARQUET:
        return _iter_parquet_rows(artifact.path)
    raise AssertionError(f"Unhandled artifact format: {artifact.format}")


def _stable_key(seed: int, source: str, question: str) -> str:
    payload = f"{seed}\x1f{source}\x1f{normalize_question(question)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def acquire_artifacts(config: Phase6Config) -> None:
    """Download every external artifact and verify its configured checksum."""
    for artifact in config.artifacts:
        download_verified(artifact.url, artifact.path, artifact.sha256)


def _load_questions(
    config: Phase6Config,
) -> tuple[list[SourceQuestion], dict[str, int], dict[str, int]]:
    by_source: dict[str, list[SourceQuestion]] = defaultdict(list)
    raw_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    for artifact in config.artifacts:
        for row in _artifact_rows(artifact):
            raw_counts[artifact.source] += 1
            try:
                by_source[artifact.source].append(_parse_ai2_row(row, artifact.source))
            except (KeyError, TypeError, ValueError):
                excluded[f"{artifact.source}:invalid_row"] += 1

    phase1_questions = {
        normalize_question(example.question)
        for example in load_dataset(config.dataset.phase1_dataset_path)
    }
    prior_phase_questions: set[str] = set()
    for path, expected_sha256 in zip(
        config.dataset.exclusion_dataset_paths,
        config.dataset.exclusion_dataset_sha256s,
        strict=True,
    ):
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Exclusion dataset checksum mismatch for {path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        prior_phase_questions.update(
            normalize_question(example.question) for example in load_dataset(path)
        )
    selected: list[SourceQuestion] = []
    seen: set[str] = phase1_questions | prior_phase_questions
    for source, questions in sorted(by_source.items()):
        unique: list[SourceQuestion] = []
        for question in sorted(
            questions,
            key=lambda item: _stable_key(config.dataset.seed, source, item.question),
        ):
            key = normalize_question(question.question)
            if key in seen:
                if key in phase1_questions:
                    reason = "phase1_overlap"
                elif key in prior_phase_questions:
                    reason = "prior_phase_overlap"
                else:
                    reason = "duplicate_question"
                excluded[f"{source}:{reason}"] += 1
                continue
            seen.add(key)
            unique.append(question)
        selected.extend(unique[: config.dataset.max_examples_per_source])
        excluded[f"{source}:source_cap"] += max(
            0, len(unique) - config.dataset.max_examples_per_source
        )
    return selected, dict(raw_counts), dict(excluded)


def _source_row(question: SourceQuestion) -> dict[str, object]:
    answers = "\n".join(f"({choice.label}) {choice.text}" for choice in question.choices)
    return {
        "base": {
            "dataset": question.source_dataset,
            "question": question.question,
            "answers": answers,
            "correct_letter": question.correct_label,
        }
    }


def write_protocol_manifest(config: Phase6Config) -> dict[str, Any]:
    """Persist the pre-registration and reject silent protocol mutation."""
    digest = protocol_sha256(config)
    path = config.protocol.manifest_path
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("protocol_sha256") != digest:
            raise ValueError(
                "Phase 6 protocol differs from the existing pre-registration; "
                "use a new experiment_id and manifest path"
            )
        return cast(dict[str, Any], existing)
    manifest: dict[str, Any] = {
        "status": "preregistered_before_fresh_test",
        "protocol_sha256": digest,
        "protocol": protocol_payload(config),
        "prior_test_policy": config.protocol.prior_test_policy
        or (
            "Phase 4 test results are considered opened and cannot serve as the "
            "confirmatory Phase 6 holdout."
        ),
    }
    write_manifest(path, manifest)
    return manifest


def build_phase6_dataset(config: Phase6Config, token_counter: TokenCounter) -> dict[str, Any]:
    """Build, deduplicate, split, and persist the fresh Phase 6 dataset."""
    protocol = write_protocol_manifest(config)
    acquire_artifacts(config)
    questions, raw_counts, external_excluded = _load_questions(config)
    source_names = tuple(sorted({question.source_dataset for question in questions}))
    examples, report = build_dataset(
        (_source_row(question) for question in questions),
        BuildConfig(
            seed=config.dataset.seed,
            allowed_datasets=source_names,
            train_fraction=config.dataset.train_fraction,
            validation_fraction=config.dataset.validation_fraction,
            test_fraction=config.dataset.test_fraction,
        ),
        token_counter,
        max_prompt_tokens=config.dataset.max_prompt_tokens,
    )
    if (
        config.dataset.expected_examples is not None
        and len(examples) != config.dataset.expected_examples
    ):
        raise ValueError(
            f"Phase 6 dataset has {len(examples)} examples; "
            f"expected exactly {config.dataset.expected_examples}"
        )
    write_dataset(config.dataset.dataset_path, examples)
    source_split_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for source in source_names:
        source_split_counts[source] = dict(
            Counter(example.split.value for example in examples if example.source_dataset == source)
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "artifacts": [
            {
                "source": artifact.source,
                "url": artifact.url,
                "sha256": artifact.sha256,
                "local_sha256": sha256_file(artifact.path),
                "format": artifact.format.value,
                "members": list(artifact.members),
            }
            for artifact in config.artifacts
        ],
        "build": {
            "seed": config.dataset.seed,
            "max_examples_per_source": config.dataset.max_examples_per_source,
            "split_fractions": {
                "train": config.dataset.train_fraction,
                "validation": config.dataset.validation_fraction,
                "test": config.dataset.test_fraction,
            },
            "phase1_dataset_sha256": sha256_file(config.dataset.phase1_dataset_path),
            "exclusion_datasets": [
                {"sha256": digest} for digest in config.dataset.exclusion_dataset_sha256s
            ],
        },
        "report": {
            "raw_counts": raw_counts,
            "source_rows_after_sampling": report.source_rows,
            "examples": report.examples,
            "split_counts": report.split_counts,
            "source_counts": report.source_counts,
            "source_split_counts": source_split_counts,
            "external_excluded_counts": external_excluded,
            "builder_excluded_counts": report.excluded_counts,
        },
        "dataset_sha256": sha256_file(config.dataset.dataset_path),
    }
    write_manifest(config.dataset.manifest_path, manifest)
    return manifest
