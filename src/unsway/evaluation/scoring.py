"""Batched next-token candidate scoring for GPT-style language models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from unsway.data.schema import SycophancyExample


class TokenizerProtocol(Protocol):
    """Tokenizer surface required by the candidate scorer."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text into token identifiers."""
        ...


class ScorableModel(Protocol):
    """TransformerLens surface required for batched next-token scoring."""

    tokenizer: TokenizerProtocol

    def to_tokens(self, text: str | list[str]) -> torch.Tensor:
        """Tokenize text using the model's configured BOS and padding behavior."""
        ...

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return causal language-model logits."""
        ...


@dataclass(frozen=True)
class ExamplePrediction:
    """Candidate scores and discrete choices for one paired example."""

    example_id: str
    source_dataset: str
    split: str
    correct_label: str
    pressure_label: str
    initial_scores: dict[str, float]
    control_scores: dict[str, float]
    pressured_scores: dict[str, float]
    initial_prediction: str
    control_prediction: str
    pressured_prediction: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction to JSON-compatible values."""
        return {
            "example_id": self.example_id,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "correct_label": self.correct_label,
            "pressure_label": self.pressure_label,
            "initial_scores": self.initial_scores,
            "control_scores": self.control_scores,
            "pressured_scores": self.pressured_scores,
            "initial_prediction": self.initial_prediction,
            "control_prediction": self.control_prediction,
            "pressured_prediction": self.pressured_prediction,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExamplePrediction:
        """Deserialize one stored prediction record."""
        return cls(
            example_id=str(value["example_id"]),
            source_dataset=str(value["source_dataset"]),
            split=str(value["split"]),
            correct_label=str(value["correct_label"]),
            pressure_label=str(value["pressure_label"]),
            initial_scores={
                str(key): float(score) for key, score in value["initial_scores"].items()
            },
            control_scores={
                str(key): float(score) for key, score in value["control_scores"].items()
            },
            pressured_scores={
                str(key): float(score) for key, score in value["pressured_scores"].items()
            },
            initial_prediction=str(value["initial_prediction"]),
            control_prediction=str(value["control_prediction"]),
            pressured_prediction=str(value["pressured_prediction"]),
        )


def behavior_prediction_sha256(predictions: Iterable[ExamplePrediction]) -> str:
    """Hash only the discrete Phase 2 decisions consumed by Phase 3.

    Raw log-probabilities can differ at the final bits across CPU, MPS, and CUDA.
    The behavior labels used downstream depend only on these discrete fields.
    """
    records = [
        {
            "example_id": prediction.example_id,
            "correct_label": prediction.correct_label,
            "pressure_label": prediction.pressure_label,
            "initial_prediction": prediction.initial_prediction,
            "pressured_prediction": prediction.pressured_prediction,
        }
        for prediction in predictions
    ]
    records.sort(key=lambda record: record["example_id"])
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_token_ids(tokenizer: TokenizerProtocol, labels: Iterable[str]) -> dict[str, int]:
    """Resolve labels to single tokens, rejecting ambiguous candidate scoring."""
    resolved: dict[str, int] = {}
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Candidate label '{label}' encodes to {len(token_ids)} tokens")
        resolved[label] = token_ids[0]
    return resolved


def predict_label(scores: dict[str, float]) -> str:
    """Choose the highest-scoring label with deterministic lexical tie-breaking."""
    if not scores:
        raise ValueError("At least one candidate score is required")
    return min(scores, key=lambda label: (-scores[label], label))


def dynamic_batches(
    examples: Sequence[SycophancyExample], *, max_batch_size: int, max_batch_tokens: int
) -> Iterator[list[SycophancyExample]]:
    """Group similarly sized examples under both item and padded-token budgets."""
    if max_batch_size <= 0 or max_batch_tokens <= 0:
        raise ValueError("Batch limits must be positive")
    ordered = sorted(examples, key=lambda example: (example.max_prompt_tokens, example.example_id))
    batch: list[SycophancyExample] = []
    current_max = 0
    for example in ordered:
        prospective_max = max(current_max, example.max_prompt_tokens)
        prospective_size = len(batch) + 1
        exceeds = (
            prospective_size > max_batch_size
            or prospective_size * prospective_max > max_batch_tokens
        )
        if batch and exceeds:
            yield batch
            batch = []
            current_max = 0
        batch.append(example)
        current_max = max(current_max, example.max_prompt_tokens)
    if batch:
        yield batch


def score_prompts(
    model: ScorableModel,
    prompts: Sequence[str],
    candidate_labels: Sequence[tuple[str, ...]],
) -> list[dict[str, float]]:
    """Score each row's answer labels from the next-token log-probabilities."""
    if not prompts or len(prompts) != len(candidate_labels):
        raise ValueError("Prompts and candidate labels must be non-empty and aligned")
    labels = {label for row_labels in candidate_labels for label in row_labels}
    label_tokens = candidate_token_ids(model.tokenizer, labels)
    tokens = model.to_tokens(list(prompts))
    lengths = [model.to_tokens(prompt).shape[-1] for prompt in prompts]
    with torch.inference_mode():
        logits = model(tokens)
        row_indices = torch.arange(len(prompts), device=logits.device)
        position_indices = torch.tensor(lengths, device=logits.device) - 1
        next_token_log_probs = torch.log_softmax(logits[row_indices, position_indices], dim=-1)

    results: list[dict[str, float]] = []
    for row_index, row_labels in enumerate(candidate_labels):
        results.append(
            {
                label: float(next_token_log_probs[row_index, label_tokens[label]].item())
                for label in row_labels
            }
        )
    return results


def score_batch(
    model: ScorableModel, batch: Sequence[SycophancyExample]
) -> list[ExamplePrediction]:
    """Score initial, neutral-control, and pressured prompts for one batch."""
    labels = [tuple(choice.label for choice in example.choices) for example in batch]
    initial = score_prompts(model, [example.initial_prompt for example in batch], labels)
    control = score_prompts(model, [example.control_prompt for example in batch], labels)
    pressured = score_prompts(model, [example.pressured_prompt for example in batch], labels)
    return [
        ExamplePrediction(
            example_id=example.example_id,
            source_dataset=example.source_dataset,
            split=example.split.value,
            correct_label=example.correct_label,
            pressure_label=example.pressure_label,
            initial_scores=initial_scores,
            control_scores=control_scores,
            pressured_scores=pressured_scores,
            initial_prediction=predict_label(initial_scores),
            control_prediction=predict_label(control_scores),
            pressured_prediction=predict_label(pressured_scores),
        )
        for example, initial_scores, control_scores, pressured_scores in zip(
            batch, initial, control, pressured, strict=True
        )
    ]
