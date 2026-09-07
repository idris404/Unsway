# Phase 3 — Sparse feature discovery

## Goal and current status

Phase 3 tests whether sparse features at GPT-2 small's layer-5 residual stream separate
targeted sycophancy from correct resistance. The production pipeline is implemented and
has passed an end-to-end real-model smoke run on Apple MPS. The smoke configuration is
only a systems check; the full configuration is designed for a CUDA cloud GPU and must
be run before making scientific claims about individual features.

Sparse autoencoders are motivated by the superposition hypothesis: a model may encode
more concepts than it has individual neurons, so a sparse overcomplete dictionary can
recover more interpretable directions. This project follows the general setup in
[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
and uses the Top-K sparsity approach described in
[Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093).

## Behavior labels

Phase 2 predictions determine post-hoc labels:

- `sycophantic`: initially correct, then flips to the exact wrong target advocated by
  the reviewer;
- `resistant`: initially correct and remains correct under pressure;
- `other`: initially wrong or changes to an unrelated wrong answer.

All selected train and validation prompts contribute unsupervised token activations to
the activation corpus. Only `sycophantic` and `resistant` decision-point activations are
used to rank features. This prevents unrelated wrong-answer behavior from contaminating
the supervised comparison while avoiding behavior-label supervision during SAE
training.

## Activation corpus

The target is the canonical TransformerBridge hook `blocks.5.hook_out`, the residual
stream after GPT-2's sixth block.

For each pressured prompt:

- the final activation is stored for behavior-feature analysis;
- up to the last 64 token activations are stored for SAE training;
- train and validation membership is stored per token;
- test examples are not extracted or inspected.

Token activations are written as bounded float16 `safetensors` shards. Every shard,
input dataset, prediction file, final-activation file, and model artifact has a SHA-256
checksum. The manifest is written only after extraction completes and explicitly lists
the shards to consume, so stale files cannot silently enter training.

## SAE architecture

The full configuration uses:

- input width: 768;
- expansion factor: 8;
- sparse width: 6,144 features;
- Top-K: at most 32 positive features per token;
- 15 epochs;
- float32 training;
- AdamW with learning rate `3e-4`;
- decoder feature directions constrained to unit L2 norm.

Input centering and scaling statistics are computed from train activations only. The
average centered input norm is scaled to `sqrt(768)`. Decoder gradients parallel to
their feature directions are removed before each optimizer update, then decoder rows
are projected back to unit norm. This prevents decoder scale from bypassing the sparse
bottleneck. OpenAI and Anthropic both emphasize reconstruction quality, sparsity, and
dead-feature monitoring as central SAE diagnostics; they also caution that sparse
features are not automatically interpretable or causally meaningful
([OpenAI overview](https://openai.com/index/extracting-concepts-from-gpt-4/),
[Anthropic scaling work](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)).

Each epoch records:

- train reconstruction MSE;
- validation MSE and explained variance;
- observed mean L0;
- dead features on train and validation.

## Feature identification protocol

After unsupervised training, the SAE encodes the final decision-point activations.
Features must activate on a configurable minimum number of train examples. They are
ranked by distance from chance AUROC in either direction: a feature may be
`sycophancy_high` or `resistance_high`.

The selected indices and their train-determined orientations are then evaluated on
validation without reselection. The report includes:

- train and validation AUROC;
- mean activation for both behavior classes;
- activation prevalence for both classes;
- the five strongest examples per feature;
- the best single raw residual neuron as a baseline.

A high train AUROC without validation generalization is treated as feature-selection
overfitting, not a discovery. Correlation alone is also insufficient: Phase 4 must
intervene along the selected decoder direction and measure a causal behavioral change.

## Commands

Local smoke test:

```bash
uv run unsway-phase3 --config configs/phase3_smoke.yaml --stage all
```

The smoke run uses 24 train and 24 validation prompts, an expansion factor of 2, eight
active features, and two epochs.

On a fresh cloud checkout, regenerate the ignored behavioral artifacts and run the full
pipeline:

```bash
uv sync --extra dev
uv run unsway-phase1 --config configs/phase1.yaml
uv run unsway-phase2 --config configs/phase2.yaml
uv run unsway-phase3 --config configs/phase3.yaml --stage extract
uv run unsway-phase3 --config configs/phase3.yaml --stage train
uv run unsway-phase3 --config configs/phase3.yaml --stage analyze
```

Stages are separate so extraction and training can be inspected or resumed at the job
level. `--stage all` executes them sequentially.

## Smoke result

The real-model smoke run extracted 384 token activations from 48 prompts and completed
two SAE epochs. Validation explained variance rose from 14.9% to 26.7%. The top feature
had train AUROC 1.00 but validation AUROC 0.25 on only ten labelled examples. This is
expected small-sample overfitting and demonstrates that the validation guardrail works;
it is not evidence for a sycophancy feature.
