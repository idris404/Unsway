"""Top-K sparse autoencoder and deterministic training loop."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as nn_functional

from unsway.data.io import write_manifest
from unsway.data.source import sha256_file
from unsway.features.config import Phase3Config
from unsway.features.corpus import iter_corpus_shards
from unsway.runtime import resolve_device, seed_everything


class TopKSparseAutoencoder(nn.Module):
    """Overcomplete autoencoder with at most K non-negative active features."""

    def __init__(self, d_in: int, d_sae: int, k: int) -> None:
        super().__init__()
        if not 0 < k <= d_sae:
            raise ValueError("K must be between 1 and the SAE width")
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        decoder = torch.randn(d_sae, d_in) / math.sqrt(d_in)
        decoder = nn_functional.normalize(decoder, dim=1)
        self.decoder_weight = nn.Parameter(decoder)
        self.encoder_weight = nn.Parameter(decoder.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
        self.decoder_bias = nn.Parameter(torch.zeros(d_in))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs and retain only their K largest positive features."""
        pre_activations = nn_functional.linear(
            inputs - self.decoder_bias, self.encoder_weight, self.encoder_bias
        )
        positive = nn_functional.relu(pre_activations)
        top_values, top_indices = torch.topk(positive, self.k, dim=-1)
        features = torch.zeros_like(positive)
        return features.scatter(-1, top_indices, top_values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct normalized model activations from sparse features."""
        return features @ self.decoder_weight + self.decoder_bias

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction and sparse feature activations."""
        features = self.encode(inputs)
        return self.decode(features), features

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder rows onto the unit sphere."""
        self.decoder_weight.copy_(nn_functional.normalize(self.decoder_weight, dim=1))

    @torch.no_grad()
    def remove_decoder_parallel_gradient(self) -> None:
        """Remove decoder gradients parallel to unit-norm feature directions."""
        gradient = self.decoder_weight.grad
        if gradient is None:
            return
        parallel = (gradient * self.decoder_weight).sum(dim=1, keepdim=True)
        gradient.sub_(parallel * self.decoder_weight)


@dataclass(frozen=True)
class NormalizationStats:
    """Affine normalization learned from training activations only."""

    mean: torch.Tensor
    scale: float

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Center and scale activations to expected norm ``sqrt(d_model)``."""
        return (values.to(torch.float32) - self.mean) * self.scale


def _split_code(manifest_path: Path, split: str) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = manifest["split_mapping"]
    if split not in mapping:
        raise ValueError(f"Corpus does not contain split '{split}'")
    return int(mapping[split])


def _iter_split_activations(manifest_path: Path, split: str) -> Iterator[torch.Tensor]:
    split_code = _split_code(manifest_path, split)
    for shard in iter_corpus_shards(manifest_path):
        mask = shard["split_codes"] == split_code
        if mask.any():
            yield shard["activations"][mask].to(torch.float32)


def compute_normalization(manifest_path: str | Path) -> NormalizationStats:
    """Compute train-only centering and expected-norm scaling statistics."""
    path = Path(manifest_path)
    total: torch.Tensor | None = None
    count = 0
    for values in _iter_split_activations(path, "train"):
        shard_sum = values.sum(dim=0, dtype=torch.float64)
        total = shard_sum if total is None else total + shard_sum
        count += values.shape[0]
    if total is None or count == 0:
        raise ValueError("Training activation corpus is empty")
    mean = (total / count).to(torch.float32)
    norm_sum = 0.0
    for values in _iter_split_activations(path, "train"):
        norm_sum += float(torch.linalg.vector_norm(values - mean, dim=1).sum().item())
    expected_norm = norm_sum / count
    if expected_norm <= 0:
        raise ValueError("Training activations have zero centered norm")
    scale = math.sqrt(mean.numel()) / expected_norm
    return NormalizationStats(mean=mean, scale=scale)


def _batches(
    values: torch.Tensor, batch_size: int, generator: torch.Generator | None
) -> Iterator[torch.Tensor]:
    order = (
        torch.arange(values.shape[0])
        if generator is None
        else torch.randperm(values.shape[0], generator=generator)
    )
    for start in range(0, values.shape[0], batch_size):
        yield values[order[start : start + batch_size]]


@torch.no_grad()
def _evaluate_split(
    model: TopKSparseAutoencoder,
    manifest_path: Path,
    split: str,
    normalization: NormalizationStats,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    squared_error = 0.0
    squared_total = 0.0
    active_counts = torch.zeros(model.d_sae, dtype=torch.int64)
    rows = 0
    for values in _iter_split_activations(manifest_path, split):
        for batch in _batches(values, batch_size, None):
            normalized = normalization.apply(batch).to(device)
            reconstruction, features = model(normalized)
            squared_error += float(((reconstruction - normalized) ** 2).sum().item())
            squared_total += float((normalized**2).sum().item())
            active_counts += (features > 0).sum(dim=0).to("cpu")
            rows += normalized.shape[0]
    if rows == 0:
        raise ValueError(f"Activation corpus split '{split}' is empty")
    return {
        "tokens": rows,
        "mse": squared_error / (rows * model.d_in),
        "explained_variance": 1.0 - squared_error / squared_total,
        "mean_l0": float(active_counts.sum().item() / rows),
        "dead_features": int((active_counts == 0).sum().item()),
    }


def train_sae(
    config: Phase3Config,
    *,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train and validate a Top-K SAE from the extracted activation corpus."""
    seed_everything(config.seed)
    device = resolve_device(config.model.device)
    manifest_path = config.output.artifact_dir / "activation_shards" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    d_in = int(manifest["d_model"])
    d_sae = d_in * config.sae.expansion_factor
    if config.sae.k > d_sae:
        raise ValueError("Configured K exceeds SAE width")
    normalization = compute_normalization(manifest_path)
    model = TopKSparseAutoencoder(d_in, d_sae, config.sae.k).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.sae.learning_rate,
        weight_decay=config.sae.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.sae.epochs + 1):
        model.train()
        error_sum = 0.0
        rows = 0
        active_counts = torch.zeros(d_sae, dtype=torch.int64)
        for values in _iter_split_activations(manifest_path, "train"):
            for batch in _batches(values, config.sae.batch_size, generator):
                normalized = normalization.apply(batch).to(device)
                reconstruction, features = model(normalized)
                loss = ((reconstruction - normalized) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                model.remove_decoder_parallel_gradient()
                optimizer.step()
                model.normalize_decoder()
                error_sum += float(loss.item()) * normalized.shape[0]
                rows += normalized.shape[0]
                active_counts += (features.detach() > 0).sum(dim=0).to("cpu")
        model.eval()
        validation = _evaluate_split(
            model,
            manifest_path,
            "validation",
            normalization,
            config.sae.batch_size,
            device,
        )
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_mse": error_sum / rows,
            "train_mean_l0": float(active_counts.sum().item() / rows),
            "train_dead_features": int((active_counts == 0).sum().item()),
            "validation": validation,
        }
        history.append(epoch_metrics)
        if progress is not None:
            progress(epoch, config.sae.epochs, epoch_metrics)

    artifact_path = config.output.artifact_dir / "sae.safetensors"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "encoder_weight": model.encoder_weight.detach().cpu(),
            "encoder_bias": model.encoder_bias.detach().cpu(),
            "decoder_weight": model.decoder_weight.detach().cpu(),
            "decoder_bias": model.decoder_bias.detach().cpu(),
            "normalization_mean": normalization.mean,
            "normalization_scale": torch.tensor(normalization.scale),
        },
        artifact_path,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "architecture": "topk",
        "d_in": d_in,
        "d_sae": d_sae,
        "k": config.sae.k,
        "epochs": config.sae.epochs,
        "batch_size": config.sae.batch_size,
        "learning_rate": config.sae.learning_rate,
        "weight_decay": config.sae.weight_decay,
        "device": str(device),
        "normalization_scale": normalization.scale,
        "history": history,
        "artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
        "corpus_manifest_sha256": sha256_file(manifest_path),
    }
    write_manifest(config.output.training_report_path, report)
    return report


def load_sae(
    path: str | Path, d_in: int, d_sae: int, k: int
) -> tuple[TopKSparseAutoencoder, NormalizationStats]:
    """Load a trained SAE and its activation normalization without pickle."""
    tensors = load_file(path)
    model = TopKSparseAutoencoder(d_in, d_sae, k)
    model.load_state_dict(
        {
            "encoder_weight": tensors["encoder_weight"],
            "encoder_bias": tensors["encoder_bias"],
            "decoder_weight": tensors["decoder_weight"],
            "decoder_bias": tensors["decoder_bias"],
        }
    )
    model.eval()
    normalization = NormalizationStats(
        mean=tensors["normalization_mean"],
        scale=float(tensors["normalization_scale"].item()),
    )
    return model, normalization
