# Phase 4 — Causal activation steering

## Goal

Phase 4 tests whether the Phase 3 correlates causally change GPT-2 small's targeted
sycophancy. The primary SAE intervention uses feature 4825. Raw residual neuron 144 is
the comparator, and a seeded random unit vector is the negative control.

The experiment does not treat validation AUROC as causal evidence. It adds each
direction to `blocks.5.hook_out` at the final non-padding token and re-scores the same
answer candidates used by the Phase 2 baseline.

## Direction conventions

All configured strengths are measured as the L2 norm of the residual-stream delta, so
the SAE, raw-neuron, and random interventions share a comparable scale.

- Feature 4825 is `sycophancy_high`, so its anti-sycophancy direction is the negative
  unit-normalized SAE decoder vector.
- Neuron 144 is `resistance_high`, so its anti-sycophancy direction is the positive raw
  residual basis vector.
- The random control is a deterministic unit vector generated from seed 4242. Its test
  strength is matched to the validation-selected SAE strength rather than optimized.

Negative doses reverse each direction and serve as a dose-response falsification
check. The intervention changes a cloned activation tensor; it never mutates cached
model state.

## Leakage-safe evaluation

Eligibility is frozen from Phase 2: an example belongs to the denominator only if the
unsteered model initially answered it correctly. Steering cannot improve its reported
rate by changing this denominator.

The protocol has two stages:

1. `sweep` evaluates signed doses on validation. For each real candidate, it chooses
   the positive dose with the lowest pressured-target sycophancy rate, provided initial
   accuracy drops by no more than five percentage points and the rate improves over
   baseline.
2. `test` reads those frozen choices and evaluates them once on the untouched test
   split. The random control receives the SAE-selected dose.

The primary metric remains the Phase 2 targeted sycophancy rate:

`P(pressured target | baseline initially correct)`

The neutral-control-adjusted pressure effect remains a key diagnostic:

`P(pressured target | baseline initially correct) - P(control target | baseline initially correct)`

Reports also contain the pressured-target rate, initial accuracy, retention of
baseline-eligible answers, any-wrong rate, and approximate paired 95% confidence
intervals for changes from baseline.

## Integrity checks

Before constructing a direction, Phase 4 verifies the dataset, discrete Phase 2
predictions, SAE artifact, Phase 3 training report, and Phase 3 feature report.
The feature and training reports must both identify the exact configured SAE checksum.

## Commands

The production Colab run is split so validation can be inspected before test is opened:

```bash
uv run unsway-phase4 --config configs/phase4.yaml --stage sweep
uv run unsway-phase4 --config configs/phase4.yaml --stage test
```

For a 24-example systems check on each split:

```bash
uv run unsway-phase4 --config configs/phase4_smoke.yaml --stage all
```

The smoke run is not a scientific result. Both configurations require the exact Phase
3 SAE artifact, which remains backed up outside Git in Google Drive.
