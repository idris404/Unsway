"""Supervised post-hoc identification of sycophancy-correlated SAE features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from unsway.data.io import write_manifest
from unsway.features.config import Phase3Config
from unsway.features.sae import load_sae
from unsway.runtime import resolve_device


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC with average ranks for ties and no external dependency."""
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Scores and labels must be aligned one-dimensional arrays")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both behavior classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _feature_statistics(
    values: np.ndarray, labels: np.ndarray, feature_index: int
) -> dict[str, float | int | str]:
    feature = values[:, feature_index]
    positive = feature[labels == 1]
    negative = feature[labels == 0]
    auroc = binary_auroc(feature, labels)
    return {
        "feature_index": feature_index,
        "auroc": auroc,
        "discrimination_auroc": max(auroc, 1.0 - auroc),
        "orientation": "sycophancy_high" if auroc >= 0.5 else "resistance_high",
        "sycophantic_mean": float(positive.mean()),
        "resistant_mean": float(negative.mean()),
        "mean_difference": float(positive.mean() - negative.mean()),
        "sycophantic_active_rate": float((positive > 0).mean()),
        "resistant_active_rate": float((negative > 0).mean()),
        "active_examples": int((feature > 0).sum()),
    }


@torch.no_grad()
def analyze_features(config: Phase3Config) -> dict[str, Any]:
    """Rank SAE latents on train labels and validate the same latents out of sample."""
    training_report = json.loads(config.output.training_report_path.read_text(encoding="utf-8"))
    extraction_report = json.loads(config.output.extraction_report_path.read_text(encoding="utf-8"))
    sae, normalization = load_sae(
        training_report["artifact"]["path"],
        int(training_report["d_in"]),
        int(training_report["d_sae"]),
        int(training_report["k"]),
    )
    device = resolve_device(config.model.device)
    sae.to(device)
    finals = load_file(extraction_report["final_activations"]["path"])
    example_metadata = json.loads(
        Path(extraction_report["example_metadata"]["path"]).read_text(encoding="utf-8")
    )["examples"]
    normalized = normalization.apply(finals["activations"])
    feature_batches: list[torch.Tensor] = []
    for start in range(0, normalized.shape[0], config.sae.batch_size):
        feature_batches.append(
            sae.encode(normalized[start : start + config.sae.batch_size].to(device)).cpu()
        )
    features = torch.cat(feature_batches).numpy()
    raw = normalized.numpy()
    labels = finals["behavior_labels"].numpy().astype(np.int64)
    split_codes = finals["split_codes"].numpy().astype(np.int64)
    split_mapping = extraction_report["split_mapping"]
    train_mask = (labels >= 0) & (split_codes == int(split_mapping["train"]))
    validation_mask = (labels >= 0) & (split_codes == int(split_mapping["validation"]))
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Feature analysis requires labelled train and validation examples")
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    validation_features = features[validation_mask]
    validation_labels = labels[validation_mask]

    candidates = []
    for feature_index in range(features.shape[1]):
        stats = _feature_statistics(train_features, train_labels, feature_index)
        if int(stats["active_examples"]) >= config.analysis.min_active_examples:
            candidates.append(stats)
    if not candidates:
        raise ValueError("No SAE features satisfy the minimum activation threshold")
    candidates.sort(
        key=lambda item: (
            float(item["discrimination_auroc"]),
            abs(float(item["mean_difference"])),
        ),
        reverse=True,
    )
    selected = candidates[: config.analysis.top_n_features]
    top_features = []
    for train_stats in selected:
        feature_index = int(train_stats["feature_index"])
        validation_stats = _feature_statistics(
            validation_features, validation_labels, feature_index
        )
        validation_auroc = float(validation_stats["auroc"])
        validation_stats["oriented_auroc"] = (
            validation_auroc
            if train_stats["orientation"] == "sycophancy_high"
            else 1.0 - validation_auroc
        )
        strongest_indices = np.argsort(features[:, feature_index], kind="mergesort")[-5:][::-1]
        strongest_examples = [
            {
                "example_id": example_metadata[index]["example_id"],
                "source_dataset": example_metadata[index]["source_dataset"],
                "split": example_metadata[index]["split"],
                "behavior_label": example_metadata[index]["behavior_label"],
                "activation": float(features[index, feature_index]),
            }
            for index in strongest_indices
        ]
        top_features.append(
            {
                "train": train_stats,
                "validation": validation_stats,
                "strongest_examples": strongest_examples,
            }
        )

    raw_train_aurocs = [
        binary_auroc(raw[train_mask, index], train_labels) for index in range(raw.shape[1])
    ]
    raw_discrimination = [max(auroc, 1.0 - auroc) for auroc in raw_train_aurocs]
    raw_index = int(np.argmax(raw_discrimination))
    raw_orientation = "sycophancy_high" if raw_train_aurocs[raw_index] >= 0.5 else "resistance_high"
    raw_validation_auroc = binary_auroc(raw[validation_mask, raw_index], validation_labels)
    report: dict[str, Any] = {
        "schema_version": 1,
        "selection_protocol": "rank_on_train_validate_without_reselection",
        "labelled_examples": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "train_sycophantic": int((train_labels == 1).sum()),
            "train_resistant": int((train_labels == 0).sum()),
            "validation_sycophantic": int((validation_labels == 1).sum()),
            "validation_resistant": int((validation_labels == 0).sum()),
        },
        "eligible_sae_features": len(candidates),
        "top_features": top_features,
        "raw_neuron_baseline": {
            "dimension": raw_index,
            "train_auroc": raw_train_aurocs[raw_index],
            "train_discrimination_auroc": raw_discrimination[raw_index],
            "orientation": raw_orientation,
            "validation_auroc": raw_validation_auroc,
            "validation_oriented_auroc": (
                raw_validation_auroc
                if raw_orientation == "sycophancy_high"
                else 1.0 - raw_validation_auroc
            ),
        },
        "sae_artifact_sha256": training_report["artifact"]["sha256"],
    }
    write_manifest(config.output.feature_report_path, report)
    return report
