"""Train-only construction of distributed Phase 6 steering directions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from unsway.data.io import write_manifest
from unsway.data.source import sha256_file
from unsway.features.analysis import binary_auroc
from unsway.features.sae import load_sae


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def load_phase6d_methods(path: str | Path) -> dict[str, Any]:
    """Load the frozen Phase 6D method specification."""
    methods_path = Path(path)
    raw = _mapping(yaml.safe_load(methods_path.read_text(encoding="utf-8")), "Phase 6D methods")
    for section in ("inputs", "construction", "selection", "output"):
        _mapping(raw.get(section), section)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported Phase 6D method schema")
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    raw["methods_sha256"] = hashlib.sha256(encoded).hexdigest()
    return raw


def write_phase6d_methods(path: str | Path) -> dict[str, Any]:
    """Version the path-independent hash of the frozen Phase 6D methods."""
    methods = load_phase6d_methods(path)
    output = _mapping(methods["output"], "output")
    report = {
        "schema_version": 1,
        "status": "frozen_before_validation",
        "methods_sha256": methods["methods_sha256"],
        "phase6_protocol_sha256": methods["phase6_protocol_sha256"],
        "phase6_dataset_sha256": methods["phase6_dataset_sha256"],
        "construction": methods["construction"],
        "selection": methods["selection"],
    }
    write_manifest(Path(str(output["methods_manifest_path"])), report)
    return report


def _unit(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float32)
    norm = torch.linalg.vector_norm(values)
    if not torch.isfinite(values).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Steering directions must be finite and non-zero")
    return cast(torch.Tensor, values / norm)


def _stratified_folds(labels: torch.Tensor, folds: int, seed: int) -> torch.Tensor:
    assignment = torch.full((labels.numel(),), -1, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    for label in (0, 1):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        assignment[indices] = torch.arange(indices.numel()) % folds
    if (assignment < 0).any():
        raise ValueError("Cross-validation labels must be binary")
    return assignment


def _caa_cv(activations: torch.Tensor, labels: torch.Tensor, folds: int, seed: int) -> float:
    assignment = _stratified_folds(labels, folds, seed)
    scores: list[float] = []
    for fold in range(folds):
        train = assignment != fold
        heldout = assignment == fold
        direction = _unit(
            activations[train & (labels == 0)].mean(0) - activations[train & (labels == 1)].mean(0)
        )
        sycophancy_scores = -(activations[heldout] @ direction).numpy()
        scores.append(binary_auroc(sycophancy_scores, labels[heldout].numpy()))
    return float(np.mean(scores))


def _validate_extraction(methods: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _mapping(methods["inputs"], "inputs")
    report = json.loads(Path(str(inputs["extraction_report_path"])).read_text(encoding="utf-8"))
    if report.get("status") != "train_validation_multilayer_extracted":
        raise ValueError("Phase 6 activation extraction is not complete")
    if report.get("protocol_sha256") != methods["phase6_protocol_sha256"]:
        raise ValueError("Phase 6D protocol checksum mismatch")
    if report.get("dataset_sha256") != methods["phase6_dataset_sha256"]:
        raise ValueError("Phase 6D dataset checksum mismatch")
    if report.get("test_examples_extracted") != 0:
        raise ValueError("Phase 6D refuses activation artifacts containing test examples")
    for key, report_key in (("activations_path", "activations"), ("metadata_path", "metadata")):
        path = Path(str(inputs[key]))
        if sha256_file(path) != report[report_key]["sha256"]:
            raise ValueError(f"Phase 6D {key} checksum mismatch")
    return report, inputs


def build_phase6d_directions(methods_path: str | Path) -> dict[str, Any]:
    """Construct CAA, legacy, SAE-composite, and matched-random directions."""
    methods = load_phase6d_methods(methods_path)
    extraction, inputs = _validate_extraction(methods)
    construction = _mapping(methods["construction"], "construction")
    output = _mapping(methods["output"], "output")
    tensors = load_file(str(inputs["activations_path"]))
    activations = tensors["activations"].to(torch.float32)
    labels = tensors["behavior_labels"].to(torch.int64)
    splits = tensors["split_codes"].to(torch.int64)
    train = (splits == 0) & (labels >= 0)
    if int((train & (labels == 0)).sum()) == 0 or int((train & (labels == 1)).sum()) == 0:
        raise ValueError("Phase 6D requires both train behavior classes")

    hooks = list(extraction["hook_points"])
    fold_count = int(construction["cv_folds"])
    seed = int(construction["seed"])
    cv_rows = []
    caa_by_layer: dict[int, torch.Tensor] = {}
    for layer in range(activations.shape[1]):
        values = activations[train, layer]
        train_labels = labels[train]
        score = _caa_cv(values, train_labels, fold_count, seed + layer)
        direction = _unit(values[train_labels == 0].mean(0) - values[train_labels == 1].mean(0))
        caa_by_layer[layer] = direction
        cv_rows.append({"layer": layer, "hook_name": hooks[layer], "mean_cv_auroc": score})
    cv_rows.sort(key=lambda row: (-float(row["mean_cv_auroc"]), int(row["layer"])))
    selected_layers = [
        int(row["layer"]) for row in cv_rows[: int(construction["selected_caa_layers"])]
    ]

    legacy_report = json.loads(
        Path(str(inputs["legacy_sae_training_report_path"])).read_text(encoding="utf-8")
    )
    legacy_path = Path(str(inputs["legacy_sae_path"]))
    expected_legacy_sha = str(inputs["legacy_sae_sha256"])
    if sha256_file(legacy_path) != expected_legacy_sha:
        raise ValueError("Legacy SAE artifact checksum mismatch")
    if legacy_report.get("artifact", {}).get("sha256") != expected_legacy_sha:
        raise ValueError("Legacy SAE report checksum mismatch")
    sae, normalization = load_sae(
        legacy_path,
        int(legacy_report["d_in"]),
        int(legacy_report["d_sae"]),
        int(legacy_report["k"]),
    )
    layer = int(construction["legacy_layer"])
    feature = int(construction["legacy_sae_feature"])
    neuron = int(construction["legacy_raw_neuron"])
    decoder = sae.decoder_weight.detach().to(torch.float32)
    legacy_direction = _unit(-decoder[feature])
    raw_direction = torch.zeros(activations.shape[-1], dtype=torch.float32)
    raw_direction[neuron] = 1.0

    layer_values = activations[train, layer]
    normalized = normalization.apply(layer_values)
    with torch.inference_mode():
        feature_values = sae.encode(normalized)
    train_labels = labels[train]
    difference = feature_values[train_labels == 1].mean(0) - feature_values[train_labels == 0].mean(
        0
    )
    top_count = int(construction["composite_top_features"])
    top_features = torch.argsort(difference.abs(), descending=True)[:top_count]
    composite = _unit((-difference[top_features, None] * decoder[top_features]).sum(0))

    rows: list[dict[str, Any]] = []
    saved: dict[str, torch.Tensor] = {}

    def add(
        name: str, family: str, hook_name: str, direction: torch.Tensor, **details: Any
    ) -> None:
        key = f"direction_{len(rows)}"
        saved[key] = _unit(direction).cpu()
        rows.append(
            {
                "name": name,
                "family": family,
                "hook_name": hook_name,
                "tensor_key": key,
                "details": details,
            }
        )

    add("sae_feature_4825", "sae_feature_4825", hooks[layer], legacy_direction, feature=feature)
    add("raw_neuron_144", "raw_neuron_144", hooks[layer], raw_direction, neuron=neuron)
    for selected_layer in selected_layers:
        add(
            f"behavioral_caa_layer_{selected_layer}",
            "behavioral_caa",
            hooks[selected_layer],
            caa_by_layer[selected_layer],
            layer=selected_layer,
        )
    add(
        "sae_composite",
        "sae_composite",
        hooks[layer],
        composite,
        features=[int(item) for item in top_features],
        weights=[float(-difference[item]) for item in top_features],
    )
    for index, real in enumerate(list(rows)):
        generator = torch.Generator().manual_seed(seed + 10_000 + index)
        add(
            f"matched_random__{real['name']}",
            "matched_random",
            str(real["hook_name"]),
            torch.randn(activations.shape[-1], generator=generator),
            matched_to=real["name"],
        )

    directions_path = Path(str(output["directions_path"]))
    directions_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(saved, directions_path)
    report = {
        "schema_version": 1,
        "status": "train_only_directions_frozen",
        "methods_sha256": methods["methods_sha256"],
        "phase6_protocol_sha256": methods["phase6_protocol_sha256"],
        "phase6_dataset_sha256": methods["phase6_dataset_sha256"],
        "extraction_report_sha256": sha256_file(Path(str(inputs["extraction_report_path"]))),
        "legacy_sae_sha256": expected_legacy_sha,
        "labelled_train_examples": int(train.sum()),
        "caa_cross_validation": cv_rows,
        "selected_caa_layers": selected_layers,
        "directions": rows,
        "artifact": {"path": str(directions_path), "sha256": sha256_file(directions_path)},
        "test_examples_used": 0,
    }
    write_manifest(Path(str(output["directions_report_path"])), report)
    return report
