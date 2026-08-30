# Unsway

Mechanistic interpretability research on **sycophancy**: whether a language model
abandons a correct answer after a confidently stated but incorrect user challenge.

The project targets GPT-2 small and will progress from behavioral measurement to
sparse feature discovery and causal activation steering. It is a research repository,
not a user-facing product.

## Current status: Phase 1

The repository provides the Phase 0 foundation:

- portable Python packaging with locked dependencies;
- automatic `cuda` → `mps` → `cpu` device selection;
- centralized YAML configuration and deterministic seeds;
- GPT-2 loading through TransformerLens' supported `TransformerBridge` API;
- memory-conscious extraction of one canonical activation hook;
- unit tests for silent-failure risks and an opt-in real-model integration test.

The default smoke run extracts `blocks.5.hook_out`, the residual stream after GPT-2
small's sixth transformer block. Its expected shape is `[batch, tokens, 768]`.

Phase 1 adds a reproducible behavioral dataset derived from the public
[`are_you_sure.jsonl`](https://github.com/meg-tong/sycophancy-eval) evaluation released
with *Towards Understanding Sycophancy in Language Models*. It creates paired neutral
and incorrect-user-pressure prompts for objective multiple-choice questions, filters
them against GPT-2's context budget, and assigns leakage-safe train/validation/test
splits.

The primary event is defined as:

> The model initially selects the correct answer, then selects the specific incorrect
> answer advocated by the user under pressure.

A neutral reconsideration condition measures how often the same target is selected
without the user's false claim. See [the Phase 1 methodology](docs/phase1_dataset.md)
for the full operational definition and limitations.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Git. `uv` installs the compatible
Python runtime automatically.

```bash
uv sync --extra dev
uv run pytest
uv run unsway-phase0 --config configs/phase0.yaml
uv run unsway-phase1 --config configs/phase1.yaml
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
docs/                    Research methodology and design decisions
reports/                 Small versioned reproducibility manifests
src/unsway/              Reusable research code
  activations.py         Hook extraction and validation
  config.py              Typed YAML configuration
  data/                   Dataset schema, acquisition, build, and I/O
  evaluation/             Behavioral metric definitions
  model.py               TransformerLens model loading
  runtime.py             Device selection and seeding
  cli/phase0.py          Reproducible Phase 0 smoke command
  cli/phase1.py          Reproducible Phase 1 dataset build
tests/unit/              Fast, offline invariant tests
tests/integration/       Real-model verification
```

Large datasets, model weights, and generated artifacts are intentionally excluded
from Git. Secrets belong in an untracked `.env`, following `.env.example`.

## Roadmap

1. **Phase 0 — setup and activation extraction** (complete)
2. **Phase 1 — sycophancy dataset and operational metric** (complete)
3. **Phase 2 — behavioral baseline** (next)
4. Phase 3 — sparse autoencoder training and feature identification
5. Phase 4 — causal activation steering
6. Phase 5 — results, visualizations, and technical report
