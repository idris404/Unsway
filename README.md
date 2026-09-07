# Unsway

Mechanistic interpretability research on **sycophancy**: whether a language model
abandons a correct answer after a confidently stated but incorrect user challenge.

The project targets GPT-2 small and will progress from behavioral measurement to
sparse feature discovery and causal activation steering. It is a research repository,
not a user-facing product.

## Current status: Phase 4 implementation

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

Phase 2 scores every candidate label directly from GPT-2's next-token
log-probabilities. On the complete dataset, GPT-2 is initially correct on 996/3,067
examples (32.47%). Among these eligible examples, it selects the user's incorrect
target in 33.43% of pressured prompts versus 27.41% of neutral-control prompts: a
paired pressure effect of **+6.02 percentage points** (approximate 95% CI:
4.47–7.58). The held-out test split shows +8.33 points. This is evidence of targeted
influence, alongside substantial general answer instability. See
[the Phase 2 baseline report](docs/phase2_baseline.md).

Phase 3 now provides a cloud-portable activation corpus and sparse-feature pipeline:
checksum-verified `safetensors` shards, a unit-normalized Top-K sparse autoencoder,
train-only feature ranking, validation-only confirmation, dead-feature diagnostics,
and top-activating example retrieval. The full CUDA run extracted 166,528 token
activations from 2,602 train/validation prompts. Its 6,144-feature Top-K SAE reached
91.31% validation explained variance with three dead validation features. Feature 4825,
selected on train only, separates sycophantic from resistant behavior with validation
AUROC 0.820; the best raw residual neuron reaches 0.830, so sparse-feature superiority
is not claimed. See
[the Phase 3 methodology](docs/phase3_sae.md).

The full CUDA run is orchestrated by the lightweight
[Phase 3 Colab notebook](notebooks/phase3_colab.ipynb), which calls the repository CLI
and backs up expensive artifacts to Google Drive.

Phase 4 adds leakage-safe causal steering at the same layer and decision position. It
sweeps signed intervention strengths on validation, freezes a dose under an accuracy
guardrail, and evaluates that dose once on test. The primary SAE direction (feature
4825) is compared with raw neuron 144 and a matched-strength random control. See the
[Phase 4 protocol](docs/phase4_steering.md).
The corresponding [Phase 4 Colab notebook](notebooks/phase4_colab.ipynb) restores the
SAE from Drive, runs validation selection, and then performs the frozen held-out test.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Git. `uv` installs the compatible
Python runtime automatically.

```bash
uv sync --extra dev
uv run pytest
uv run unsway-phase0 --config configs/phase0.yaml
uv run unsway-phase1 --config configs/phase1.yaml
uv run unsway-phase2 --config configs/phase2.yaml
uv run unsway-phase3 --config configs/phase3_smoke.yaml --stage all
uv run unsway-phase4 --config configs/phase4_smoke.yaml --stage all
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
  features/               Activation corpus, Top-K SAE, feature analysis
  steering/               Causal directions, dose selection, held-out evaluation
  model.py               TransformerLens model loading
  runtime.py             Device selection and seeding
  cli/phase0.py          Reproducible Phase 0 smoke command
  cli/phase1.py          Reproducible Phase 1 dataset build
  cli/phase2.py          Batched Phase 2 behavioral baseline
  cli/phase3.py          Phase 3 extract/train/analyze pipeline
tests/unit/              Fast, offline invariant tests
tests/integration/       Real-model verification
```

Large datasets, model weights, and generated artifacts are intentionally excluded
from Git. Secrets belong in an untracked `.env`, following `.env.example`.

## Roadmap

1. **Phase 0 — setup and activation extraction** (complete)
2. **Phase 1 — sycophancy dataset and operational metric** (complete)
3. **Phase 2 — behavioral baseline** (complete)
4. **Phase 3 — sparse autoencoder training and feature identification** (pipeline ready; full CUDA run pending)
5. Phase 4 — causal activation steering
6. Phase 5 — results, visualizations, and technical report
