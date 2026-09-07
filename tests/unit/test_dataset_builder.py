"""Tests for deterministic paired-dataset construction."""

from unsway.data.builder import build_dataset
from unsway.data.config import BuildConfig
from unsway.data.schema import DatasetSplit
from unsway.data.source import iter_jsonl

CONFIG = BuildConfig(
    seed=42,
    allowed_datasets=("mmlu_mc_cot", "aqua_mc"),
    train_fraction=0.5,
    validation_fraction=0.25,
    test_fraction=0.25,
)


def character_counter(text: str) -> int:
    """Cheap deterministic stand-in for a tokenizer."""
    return len(text)


def test_builds_objective_pressure_pairs_and_reports_exclusions() -> None:
    """The builder creates wrong-answer interventions and removes leakage risks."""
    examples, report = build_dataset(
        iter_jsonl("tests/fixtures/source_rows.jsonl"),
        CONFIG,
        character_counter,
        max_prompt_tokens=1000,
    )

    assert len(examples) == 2
    assert report.excluded_counts == {
        "unsupported_dataset": 1,
        "duplicate_question": 1,
    }
    for example in examples:
        assert example.pressure_label != example.correct_label
        assert f"Answer: ({example.correct_label})" in example.pressured_prompt
        pressure_text = next(
            choice.text for choice in example.choices if choice.label == example.pressure_label
        )
        assert f'correct choice is "{pressure_text}"' in example.pressured_prompt
        assert "I am certain" not in example.control_prompt


def test_build_is_deterministic() -> None:
    """IDs, pressure labels, order, and partitions are seed-stable."""
    first, _ = build_dataset(
        iter_jsonl("tests/fixtures/source_rows.jsonl"),
        CONFIG,
        character_counter,
        max_prompt_tokens=1000,
    )
    second, _ = build_dataset(
        iter_jsonl("tests/fixtures/source_rows.jsonl"),
        CONFIG,
        character_counter,
        max_prompt_tokens=1000,
    )

    assert first == second
    assert {example.split for example in first} <= set(DatasetSplit)


def test_excludes_prompts_over_context_budget() -> None:
    """Oversized prompts are removed before model inference."""
    examples, report = build_dataset(
        iter_jsonl("tests/fixtures/source_rows.jsonl"),
        CONFIG,
        character_counter,
        max_prompt_tokens=10,
    )

    assert examples == []
    assert report.excluded_counts["context_too_long"] == 2
