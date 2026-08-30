# Unsway

Mechanistic interpretability research on **sycophancy**: whether a language model
abandons a correct answer after a confidently stated but incorrect user challenge.

The project targets GPT-2 small and will progress from behavioral measurement to
sparse feature discovery and causal activation steering. It is a research repository,
not a user-facing product.

## Current status: Phase 0

The repository currently provides:

- portable Python packaging with locked dependencies;
- automatic `cuda` → `mps` → `cpu` device selection;
- centralized YAML configuration and deterministic seeds;
- GPT-2 loading through TransformerLens' supported `TransformerBridge` API;
- memory-conscious extraction of one canonical activation hook;
- unit tests for silent-failure risks and an opt-in real-model integration test.

The default smoke run extracts `blocks.5.hook_out`, the residual stream after GPT-2
small's sixth transformer block. Its expected shape is `[batch, tokens, 768]`.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Git. `uv` installs the compatible
Python runtime automatically.

```bash
uv sync --extra dev
uv run pytest
uv run unsway-phase0 --config configs/phase0.yaml
```

The first smoke run downloads GPT-2 small from Hugging Face. To run the explicit
real-model integration test:

```bash
uv run pytest -m integration
```

Run all local quality gates with:

```bash
make check
```

## Repository layout

```text
configs/                 Versioned experiment settings
src/unsway/              Reusable research code
  activations.py         Hook extraction and validation
  config.py              Typed YAML configuration
  model.py               TransformerLens model loading
  runtime.py             Device selection and seeding
  cli/phase0.py          Reproducible Phase 0 smoke command
tests/unit/              Fast, offline invariant tests
tests/integration/       Real-model verification
```

Large datasets, model weights, and generated artifacts are intentionally excluded
from Git. Secrets belong in an untracked `.env`, following `.env.example`.

## Roadmap

1. **Phase 0 — setup and activation extraction** (current)
2. Phase 1 — sycophancy dataset and operational metric
3. Phase 2 — behavioral baseline
4. Phase 3 — sparse autoencoder training and feature identification
5. Phase 4 — causal activation steering
6. Phase 5 — results, visualizations, and technical report

