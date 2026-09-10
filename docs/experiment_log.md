# Experiment log

This file records the decisions needed to audit or reproduce Unsway. The scientific
narrative and interpretation live in the [technical report](technical_report.md).

## Phase 1 — dataset

The source is `datasets/are_you_sure.jsonl` from `meg-tong/sycophancy-eval`, pinned at
commit `9a1694221e3639887138f61deae344335eca6752`. The downloaded file is checked against
SHA-256 `16e034c2ec6a6145c0058863a7c0f41fee5ffa7f9f0391547ae3685e713f115f`.

Only objective AQuA, math, MMLU and TruthfulQA multiple-choice examples are retained.
Questions are deduplicated, filtered to fit GPT-2's context and assigned deterministic
70/15/15 train, validation and test splits.

Each example contains three aligned prompts: initial answer, neutral reconsideration
and false pressure. The advocated wrong answer is selected by a seed-stable SHA-256
mapping. Its label is omitted from the pressure text to reduce lexical label copying.

The primary event is selection of that specific wrong target under pressure,
conditional on the model having answered the initial prompt correctly. The matched
neutral-target rate controls for general answer instability.

```bash
uv run unsway-phase1 --config configs/phase1.yaml
```

Output: [`reports/phase1_manifest.json`](../reports/phase1_manifest.json).

## Phase 2 — behavioral baseline

Every available answer label is verified to occupy one GPT-2 token and scored from its
next-token log-probability. This avoids sampling and free-generation formatting
failures. Length-sorted dynamic batches bound both example and padded-token counts.

An early run was discarded after revealing a label-priming confound: the pressure
prompt explicitly repeated the wrong label. The final dataset advocates only the
answer text and has a new checksum.

The complete float32 MPS run evaluated 3,067 examples in 251.4 seconds. GPT-2 answered
996 correctly. The pressured target rate was 33.43%, the neutral target rate 27.41%,
and the paired effect +6.02 points (approximate 95% CI: +4.47 to +7.58). The effect was
positive in every split and source subset.

```bash
uv run unsway-phase2 --config configs/phase2.yaml
```

Output: [`reports/phase2_baseline.json`](../reports/phase2_baseline.json).

## Phase 3 — sparse feature discovery

The production T4 run extracted 166,528 token activations from 2,602 train and
validation prompts at `blocks.5.hook_out`. Test examples were not inspected. Token
activations were stored as bounded float16 `safetensors` shards with checksums.

The Top-K SAE uses 768 inputs, 6,144 latents, `k=32`, float32 training and unit-norm
decoder directions. Train-only centering and scaling prevent validation leakage.
After 15 epochs, validation explained variance reached 91.31%, mean L0 was 32 and
three validation features were dead.

Feature 4825 was ranked first using train behavior labels and reached validation AUROC
0.820 without reselection. Raw neuron 144 reached 0.830, so sparse-feature superiority
is not supported.

```bash
uv run unsway-phase3 --config configs/phase3.yaml --stage extract
uv run unsway-phase3 --config configs/phase3.yaml --stage train
uv run unsway-phase3 --config configs/phase3.yaml --stage analyze
```

Outputs: [`phase3_extraction.json`](../reports/phase3_extraction.json),
[`phase3_training.json`](../reports/phase3_training.json) and
[`phase3_features.json`](../reports/phase3_features.json). The corresponding
[Colab notebook](../notebooks/phase3_colab.ipynb) restores and backs up large
artifacts through Google Drive.

## Phase 4 — causal steering

Unit-normalized directions are added to the same layer at the final non-padding token.
Signed doses are swept on validation. A positive dose is selected only if targeted
sycophancy falls without more than five points of initial-accuracy loss. The dose is
then frozen before test evaluation. Eligibility always comes from the unsteered
baseline, so steering cannot improve the metric by changing its denominator.

SAE feature 4825 did not improve the validation rate at any positive dose. Raw neuron
144 at `+4` changed validation from 34/147 to 32/147 and was selected. On the untouched
test split it changed 45/144 to 44/144: -0.69 points, with an approximate 95% CI of
-2.06 to +0.67. Initial accuracy stayed at 30.97%.

Because the interval includes zero, this is an inconclusive causal result. It does not
show that the intervention reliably reduces sycophancy.

```bash
uv run unsway-phase4 --config configs/phase4.yaml --stage sweep
uv run unsway-phase4 --config configs/phase4.yaml --stage test
```

Outputs: [`phase4_validation.json`](../reports/phase4_validation.json) and
[`phase4_test.json`](../reports/phase4_test.json). The production workflow is in the
[Phase 4 Colab notebook](../notebooks/phase4_colab.ipynb).

## Phase 5 — reporting

The final command reads the versioned reports and regenerates five SVG figures plus a
compact JSON summary. It does not require the model, datasets or a GPU.

```bash
uv run unsway-phase5 --config configs/phase5.yaml
```

Output: [`reports/phase5_summary.json`](../reports/phase5_summary.json).

## Phase 6 — fresh-holdout extension (in progress)

Phase 6 tests whether distributed contrastive or multi-feature directions can improve
on the inconclusive single-direction result. The Phase 4 test is explicitly treated as
opened and cannot serve as this extension's confirmatory holdout.

The pre-registered dataset combines 2,000 deterministic samples from each of
CommonsenseQA, OpenBookQA and AI2 ARC. Exact normalized overlaps with Phase 1 and
within-source duplicates are removed before a stratified 60/20/20 split. The resulting
6,000 examples contain 3,600 train, 1,200 validation and 1,200 unopened test prompts.

The protocol freezes all 12 GPT-2 residual layers, four real direction families, a
matched random control, dose values, a minimum of 300 initially-correct test examples,
the pressure-specific difference-in-differences metric and success guardrails before
test evaluation.

The initial Phase 6A pre-registration had SHA-256
`cda7e8d58b32c91e1519c38706c06577d40bbd8f86b5443ac7b6453740e3930a`. Before any
production model run, v2 amended it to include CUDA, float32 model inference, float16
activation storage and exact batch limits in the hash. The amended protocol explicitly
references v1 and has SHA-256
`515d99b1d006e12f2ad5b80ee2ec41130d2845bb7852635ac61357c6f7f7114f`.

```bash
uv run unsway-phase6 --config configs/phase6.yaml --stage data
```

Current outputs: [`phase6_protocol_v2.json`](../reports/phase6_protocol_v2.json) and
[`phase6_dataset_v2.json`](../reports/phase6_dataset_v2.json). The v1 manifests remain
versioned as an audit trail.

Phase 6B has a leakage-safe behavioral runner and a batched multicouche extractor. The
runner scores all three conditions for train/validation but serializes only initial
predictions for test. Extraction is blocked unless the 300-example eligibility
guardrail passes, and it rejects test examples by construction. The production entry
point is [`phase6b_colab.ipynb`](../notebooks/phase6b_colab.ipynb).

The production baseline stopped at the pre-registered guardrail: GPT-2 answered
299/1,200 test questions correctly before pressure, one short of the required 300.
The Phase 6 v2 pressure and control prompts were never scored, and multicouche
extraction was not run. This is an insufficient-eligibility outcome, not a steering
result. The exact observed summary is versioned in
[`phase6b_outcome.json`](../reports/phase6b_outcome.json); the full Colab report still
needs to be retrieved.

## Phase 6C — disjoint replacement holdout and gated extraction

Phase 6C preserves the 300-example threshold and every steering success criterion.
It retires all 6,000 Phase 6 v2 questions and deterministically samples a new,
test-only holdout from the unused rows of the same checksum-pinned sources. The new
holdout contains 2,400 examples: 800 each from CommonsenseQA, OpenBookQA and AI2 ARC.
It has zero normalized question overlap with Phase 1 or Phase 6 v2.

The v3 protocol explicitly amends v2 and records why replacement was necessary. Its
SHA-256 is `9857c4b22c4555c78f1361fcf7a14eda8a89270924a8cd7bb5a016566138ec9a`;
the replacement dataset SHA-256 is
`264ef3afa5f2257956b6f8ddb4e01c35032fd08bf56ebda7041dd82ff347acd4`.
Only initial prompts may be evaluated until at least 300 initially correct examples
are confirmed.

```bash
uv run unsway-phase6 --config configs/phase6c.yaml --stage data
```

Current outputs: [`phase6_protocol_v3.json`](../reports/phase6_protocol_v3.json) and
[`phase6c_dataset.json`](../reports/phase6c_dataset.json). The one-shot initial-only
production run is prepared in
[`phase6c_colab.ipynb`](../notebooks/phase6c_colab.ipynb).

The production initial-only run passed the frozen eligibility gate with 537/2,400
correct answers (22.375%), versus the required 300. By source, GPT-2 answered 186/800
AI2 ARC, 152/800 CommonsenseQA and 199/800 OpenBookQA questions correctly. Neither the
pressure nor control condition was scored on this replacement holdout.

The validated gate unlocked extraction from the already-opened Phase 6 v2 train and
validation splits only. The production tensor has shape `[4800, 12, 768]`: 3,600 train
and 1,200 validation examples across all 12 GPT-2 residual layers. Its behavior labels
contain 288 sycophantic, 113 resistant and 4,399 other examples. The extractor recorded
zero test examples and verified the replacement protocol, dataset and initial-prediction
checksums before running. The compact production summary is versioned in
[`phase6c_outcome.json`](../reports/phase6c_outcome.json); large tensor artifacts remain
outside Git.
