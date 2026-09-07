# Phase 2 — GPT-2 small behavioral baseline

## Objective

Phase 2 measures whether GPT-2 small's answer distribution moves toward an incorrect
answer advocated by a confident reviewer. It establishes the fixed behavioral baseline
that later activation steering must improve or worsen.

## Scoring method

Each dataset prompt ends immediately after `Answer: (` or `Revised answer: (`. Every
available answer label is verified to encode as exactly one GPT-2 token. The model runs
once per prompt condition, and the label with the largest next-token log-probability is
the prediction.

This constrained scoring avoids treating formatting failures or verbose completions as
factual mistakes. It is also deterministic: sampling temperature and generation
parameters do not enter the experiment.

Three forward passes are evaluated per example:

1. initial answer;
2. answer after neutral reconsideration;
3. answer after confident advocacy of an incorrect answer choice.

The evaluator uses length-sorted dynamic batches. Both the number of examples and the
padded-token count are bounded, preventing the full-vocabulary logits from exhausting
MPS or CUDA memory on long prompts.

## Label-priming control

An initial full run exposed a prompt confound: explicitly writing both `(B)` and
`not (A)` before asking for a new label made the most recent label a plausible lexical
copy target. That run was discarded. The final pressure prompt advocates only the
incorrect answer's text, without repeating its label. The dataset was rebuilt with a
new checksum before the final baseline was run.

This decision makes a change toward the target require at least a mapping from the
advocated answer text back to its choice label, rather than direct label copying.

## Results

The final run used GPT-2 small in float32 on Apple MPS and evaluated all 3,067 examples
in 251.4 seconds.

- Initial accuracy: 996/3,067 = **32.47%**.
- Pressured target rate, conditional on initial correctness: 333/996 = **33.43%**.
- Neutral-control target rate on the same eligible population: **27.41%**.
- Paired pressure effect: **+6.02 percentage points**.
- Approximate 95% confidence interval for the paired effect: **+4.47 to +7.58 points**.
- Flip from correct to any incorrect answer under pressure: **61.04%**.

The pressure effect has the same positive direction in every split:

- train: +5.67 points over 705 eligible examples;
- validation: +5.44 points over 147 eligible examples;
- test: +8.33 points over 144 eligible examples.

It is also positive in every source subset, ranging from +1.67 points on TruthfulQA MC
to +9.45 points on the math subset.

The confidence interval for a single rate uses the Wilson score interval. The pressure
effect interval uses the standard error of the paired per-example difference between
the pressured-target and neutral-target indicators.

## Interpretation

The positive paired effect supports a narrow claim: confident incorrect advocacy makes
GPT-2 more likely to select that specific wrong answer than a neutral request to check
its work.

The raw 33.43% pressured target rate must not be presented alone. GPT-2 already selects
the same randomly assigned wrong target in 27.41% of neutral controls, partly because
of answer-label preferences and general instability. The pressure-specific estimate is
therefore the paired +6.02-point difference.

The 61.04% any-wrong flip rate shows that the model is highly unstable under this
completion format. Sycophancy is one component of that instability, not a complete
explanation for it.

## Reproducibility

Run:

```bash
uv run unsway-phase2 --config configs/phase2.yaml
```

The committed aggregate report is `reports/phase2_baseline.json`. Per-example logits
and predictions are generated under `data/processed/` and excluded from Git. The
report records both the Phase 1 dataset checksum and the prediction-file checksum.

## Implications for Phase 3

Feature discovery should use train examples that contrast:

- initially correct trials that flip specifically to the pressured target;
- initially correct trials that remain correct under the same pressure.

Neutral-control activations should remain available as a confound check. Otherwise, a
probe or SAE feature could capture generic reconsideration, answer instability, prompt
length, or source-dataset identity rather than targeted sycophancy.

