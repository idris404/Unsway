"""Tests for next-token scoring and memory-bounded batching."""

from typing import Any

import pytest
import torch

from unsway.data.schema import Choice, DatasetSplit, SycophancyExample
from unsway.evaluation.scoring import (
    ExamplePrediction,
    behavior_prediction_sha256,
    candidate_token_ids,
    dynamic_batches,
    predict_label,
    score_prompts,
)


class FakeTokenizer:
    """Minimal character tokenizer with single-token answer labels."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text == "MULTI":
            return [1, 2]
        if len(text) == 1 and text.isalpha():
            return [ord(text.upper()) - ord("A") + 1]
        return list(range(10, 10 + len(text)))


class FakeScorableModel:
    """Deterministic causal model whose A score grows with sequence position."""

    tokenizer = FakeTokenizer()

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        texts = [text] if isinstance(text, str) else text
        width = max(len(item) for item in texts)
        return torch.zeros((len(texts), width), dtype=torch.long)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, positions = tokens.shape
        logits = torch.zeros((batch, positions, 32), dtype=torch.float32)
        for position in range(positions):
            logits[:, position, 1] = float(position)
            logits[:, position, 2] = -float(position)
        return logits


def test_behavior_prediction_checksum_ignores_scores_and_order() -> None:
    """Cross-device float noise does not invalidate identical behavior labels."""

    def make_prediction(example_id: str, score: float) -> ExamplePrediction:
        return ExamplePrediction(
            example_id=example_id,
            source_dataset="source",
            split="train",
            correct_label="A",
            pressure_label="B",
            initial_scores={"A": score},
            control_scores={"A": score},
            pressured_scores={"B": score},
            initial_prediction="A",
            control_prediction="A",
            pressured_prediction="B",
        )

    first = [make_prediction("2", 0.1), make_prediction("1", 0.2)]
    second = [make_prediction("1", 9.0), make_prediction("2", -4.0)]

    assert behavior_prediction_sha256(first) == behavior_prediction_sha256(second)


def make_example(identifier: str, tokens: int) -> SycophancyExample:
    """Create a minimal example for batching tests."""
    return SycophancyExample(
        example_id=identifier,
        source_dataset="test",
        split=DatasetSplit.TEST,
        question="Question?",
        choices=(Choice("A", "right"), Choice("B", "wrong")),
        correct_label="A",
        pressure_label="B",
        initial_prompt="x",
        control_prompt="xx",
        pressured_prompt="xxx",
        max_prompt_tokens=tokens,
    )


def test_candidate_labels_must_be_single_tokens() -> None:
    """The baseline refuses incomparable multi-token answer labels."""
    tokenizer = FakeTokenizer()

    assert candidate_token_ids(tokenizer, ["A", "B"]) == {"A": 1, "B": 2}
    with pytest.raises(ValueError, match="encodes to 2 tokens"):
        candidate_token_ids(tokenizer, ["MULTI"])


def test_score_prompts_uses_each_unpadded_final_position() -> None:
    """Right padding cannot change which causal position is scored."""
    scores = score_prompts(
        FakeScorableModel(),
        ["xx", "xxxxx"],
        [("A", "B"), ("A", "B")],
    )

    assert scores[0]["A"] > scores[0]["B"]
    assert scores[1]["A"] > scores[1]["B"]
    assert scores[1]["A"] > scores[0]["A"]


def test_dynamic_batches_respect_item_and_token_budgets() -> None:
    """Every emitted batch stays within the configured padded-token budget."""
    examples = [make_example(str(index), tokens) for index, tokens in enumerate([5, 5, 8, 8, 9])]
    batches = list(dynamic_batches(examples, max_batch_size=3, max_batch_tokens=16))

    flattened: list[Any] = [example for batch in batches for example in batch]
    assert {example.example_id for example in flattened} == {str(index) for index in range(5)}
    for batch in batches:
        assert len(batch) <= 3
        assert len(batch) * max(example.max_prompt_tokens for example in batch) <= 16


def test_predict_label_breaks_exact_ties_lexically() -> None:
    """Prediction output is deterministic even under an exact logit tie."""
    assert predict_label({"B": -1.0, "A": -1.0}) == "A"
