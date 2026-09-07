"""Tests for Top-K sparse autoencoder invariants."""

import torch

from unsway.features.sae import TopKSparseAutoencoder


def test_topk_sparsity_and_shapes() -> None:
    """Encoding is non-negative and never activates more than K latents."""
    model = TopKSparseAutoencoder(d_in=4, d_sae=12, k=3)
    inputs = torch.randn(5, 4)

    reconstruction, features = model(inputs)

    assert reconstruction.shape == inputs.shape
    assert features.shape == (5, 12)
    assert torch.all(features >= 0)
    assert torch.all((features > 0).sum(dim=1) <= 3)


def test_decoder_projection_restores_unit_feature_directions() -> None:
    """Decoder scale cannot bypass the sparsity constraint."""
    model = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    with torch.no_grad():
        model.decoder_weight.mul_(3.0)

    model.normalize_decoder()

    assert torch.allclose(
        torch.linalg.vector_norm(model.decoder_weight, dim=1),
        torch.ones(8),
        atol=1e-6,
    )


def test_one_training_step_updates_parameters_with_finite_loss() -> None:
    """The gradient projection and unit-norm update compose correctly."""
    model = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(16, 4)
    reconstruction, _ = model(inputs)
    loss = ((reconstruction - inputs) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    model.remove_decoder_parallel_gradient()
    optimizer.step()
    model.normalize_decoder()

    assert torch.isfinite(loss)
    assert torch.allclose(
        torch.linalg.vector_norm(model.decoder_weight, dim=1),
        torch.ones(8),
        atol=1e-6,
    )
