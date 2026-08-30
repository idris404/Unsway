"""Validated schema for paired sycophancy examples."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast


class DatasetSplit(StrEnum):
    """Leakage-safe experimental partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class Choice:
    """One labelled answer choice."""

    label: str
    text: str

    def to_dict(self) -> dict[str, str]:
        """Serialize this choice to JSON-compatible values."""
        return {"label": self.label, "text": self.text}


@dataclass(frozen=True)
class SycophancyExample:
    """One objective question with neutral and pressured counterfactual prompts."""

    example_id: str
    source_dataset: str
    split: DatasetSplit
    question: str
    choices: tuple[Choice, ...]
    correct_label: str
    pressure_label: str
    initial_prompt: str
    control_prompt: str
    pressured_prompt: str
    max_prompt_tokens: int

    def __post_init__(self) -> None:
        labels = [choice.label for choice in self.choices]
        if len(labels) < 2 or len(labels) != len(set(labels)):
            raise ValueError("Examples require at least two uniquely labelled choices")
        if self.correct_label not in labels:
            raise ValueError("Correct label is absent from choices")
        if self.pressure_label not in labels:
            raise ValueError("Pressure label is absent from choices")
        if self.pressure_label == self.correct_label:
            raise ValueError("Pressure label must be objectively incorrect")
        if self.max_prompt_tokens <= 0:
            raise ValueError("Prompt token count must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the example to a stable JSON-compatible mapping."""
        return {
            "example_id": self.example_id,
            "source_dataset": self.source_dataset,
            "split": self.split.value,
            "question": self.question,
            "choices": [choice.to_dict() for choice in self.choices],
            "correct_label": self.correct_label,
            "pressure_label": self.pressure_label,
            "initial_prompt": self.initial_prompt,
            "control_prompt": self.control_prompt,
            "pressured_prompt": self.pressured_prompt,
            "max_prompt_tokens": self.max_prompt_tokens,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SycophancyExample:
        """Deserialize and validate an example from JSON-compatible values."""
        raw_choices = cast(list[dict[str, Any]], value["choices"])
        return cls(
            example_id=str(value["example_id"]),
            source_dataset=str(value["source_dataset"]),
            split=DatasetSplit(str(value["split"])),
            question=str(value["question"]),
            choices=tuple(
                Choice(label=str(choice["label"]), text=str(choice["text"]))
                for choice in raw_choices
            ),
            correct_label=str(value["correct_label"]),
            pressure_label=str(value["pressure_label"]),
            initial_prompt=str(value["initial_prompt"]),
            control_prompt=str(value["control_prompt"]),
            pressured_prompt=str(value["pressured_prompt"]),
            max_prompt_tokens=int(value["max_prompt_tokens"]),
        )
