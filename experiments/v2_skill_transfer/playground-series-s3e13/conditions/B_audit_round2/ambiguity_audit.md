# Ambiguity Audit — Playground Series S3E13 (Vector-Borne Disease, MPA@3)

Auditor stance: adversarial. Findings below are specific to the artifacts in `data/` as of this read; the spec disagrees with the files in several material ways. Solver must defend every choice they make to bridge those gaps.

## 1. Target column & class space

- **Target**: `prognosis` (string). Train has it, test does not. This is the only schema diff: `set(train.columns) − set(test.columns) = {'prognosis'}`; no extra columns sneak into test.
- **Plausible distractors** the solver might mis-pick as target if they don't read carefully: `bullseye_rash` (last symptom column, name suggests "outcome"), `microcephaly`, `paralysis`, `coma` — all are binary symptom features, not the label. Spec text lists `prognosis` last in its column-name dump, which is easy to miss. Solver must confirm `prognosis` explicitly.
- **n_classes = 11** in train, all labeled, no NaN target. Class distribution is mildly imbalanced but not pathological (37–67 per class, ratio ~1.8x): West_Nile_fever 67, Japanese_encephalitis 64, Tungiasis 58, Rift_Valley_fever 58, Dengue 57, Chikungunya 54, Yellow_Fever 46, Zika 45, Lyme_disease 41, Malaria 38, Plague 37.
- **Test never contains an unseen label** — this is the closed-set assumption the solver should state. If they invent novel labels (e.g. clustering), they auto-score 0 on every row.
- **Label strings** all use underscores already (e.g. `West_Nile_fever`, `Japanese_encephalitis`, `Yellow_Fever`). None contain spaces in train. Spec phrasing "spaces have been replaced with underscores" is descriptive of the source transformation, not an action the solver needs to perform.

## 2. MPA@3 gaming surfaces

- **Always submit 3.** A single-prediction strategy can score at most 1.0 per row but throws away two free shots. With 11 classes and a reasonable classifier, P(true in top-3) >> P(true == argmax); even random top-3 yields ~3/11 ≈ 0.27 vs. random top-1 ~0.09. Spec literally says "3 maximizes expected MPA@3" — anyone submitting fewer than 3 needs a written justification.
- **Spec is ambiguous on the scoring curve.** Two contradictory claims in the prompt:
  1. "the earlier a correct prediction occurs, the higher the score" (implies position-weighted, e.g. MAP@3 = 1, 1/2, 1/3)
  2. "The score for a row is 1 if the true label appears anywhere in the predicted top-3, else 0" (implies plain hit@3, position-agnostic)
  These are different metrics. Solver must declare which interpretation they target. Under interpretation (1), ranking matters — put highest-probability class first. Under (2), ranking is moot. **Default to (1)** because the metric name "MPA@3" matches Kaggle's MAP@3 convention; the position-agnostic phrasing is most likely a spec error. The solver should still order outputs by `predict_proba` descending — it is dominated by no other order under either interpretation.
- **Top-3 by `predict_proba` is correct** as a single-row optimum under either interpretation. Diversity-based reranking (e.g. covering distinct disease families) only helps if you assume the marginal class distribution is mis-calibrated; the solver must justify any deviation from raw argsort. Beware: scikit's `classifier.classes_` order is alphabetical/lexicographic, not training-frequency order — easy off-by-one bug.
- **Whitespace-split safety**: I verified `'West_Nile_fever Japanese_encephalitis Rift_Valley_fever'.split()` returns the correct 3-token list. Underscores survive. **Failure mode to flag**: any solver who "prettifies" labels by replacing underscores with spaces before joining will blow up the eval (`'Rift Valley fever Yellow Fever Dengue'.split()` → 6 tokens). Do not transform label strings on output.
- **Duplicate predictions in a row**: spec is silent on whether `"Dengue Dengue Zika"` counts the same as `"Dengue Zika"`. Under hit@3 it doesn't matter; under MAP@3 it wastes a slot. Solver should output 3 distinct labels.
- **CSV quoting**: the `prognosis` field contains a space-separated string — written naively with `pandas.to_csv`, pandas will NOT quote it because there is no comma; this is fine. Verify by round-tripping their own submission and re-reading with `pd.read_csv`.

## 3. Spec claims vs. reality — VERIFICATION FAILURES

This section lists places where the prompt is factually wrong about the files shipped:

- **Row counts off-by-one.** Spec: train=566, test=143. Actual: **train=565, test=142**. Solver should report actuals, not parrot the spec.
- **`sample_submission.csv` does NOT align with `test.csv`.** This is the biggest landmine:
  - `sample_submission` has **303 rows**, ids **707..1009**.
  - `test.csv` has **142 rows**, ids **2..703**.
  - `set(sub.id) ∩ set(test.id) = ∅` — they are entirely disjoint.
  - The sample file appears to be the original Kaggle public sample (for a larger leaderboard split) and is not usable as a template for this task's test set.
  - **The solver MUST emit one row per `test.csv` id (142 rows), not 303.** Anyone who blindly fills `sample_submission` will submit predictions for ids that don't exist in test and will be missing predictions for every real test id.
  - **`id` column matches `test.id` exactly** (cardinality 142, range 2..703, no overlap with train ids 0..706). Train and test ids are disjoint.
- **Sample submission's predicted values** are all `"Dengue Zika Japanese_encephalitis"` — confirms the expected output shape (space-separated, 3 tokens, underscores preserved). At least that contract is honored.
- **Schema integrity**: 64 binary symptom features, all `float64`, values strictly in {0.0, 1.0}, **zero NaN in both train and test**. Spec's hedge "Examples of columns include..." understates the schema; the solver should enumerate all 64.

## 4. Other interpretive gaps & traps

- **Synthetic-but-near-original** data per spec. The solver may not augment with the original UCI/Kaggle vector-borne-disease dataset without flagging it as an out-of-distribution leak; the spec doesn't explicitly forbid it but doesn't grant it either. Default to in-file training only.
- **Small n.** 565 train rows / 11 classes ≈ 51 per class. Cross-validation must be **stratified k-fold** (e.g., StratifiedKFold(5)) — a plain split could lose classes from a fold. Hyperparameter search must use the same CV scheme or a held-out stratified split, not the leaderboard.
- **No NaN, no missingness story.** Spec mentions nothing about missingness because there is none. Solver should not write imputation code "just in case" — it's dead weight.
- **Feature semantics not given.** All 64 columns are presence/absence of clinical symptoms (`headache`, `mouth_bleed`, ...). The prompt provides no codebook; solver cannot validate that 1 means "present". I confirmed: 0/1 only, mean row-sum ~20 symptoms per patient in both train and test (very similar distributions, so no obvious train/test drift on simple summary stats).
- **No duplicate rows in train, no train/test feature collision check is given.** The solver should at minimum check that test rows don't exactly match train rows; if many do, kNN on Hamming distance is a strong baseline and any "fancy model" needs to beat it.
- **Random seed**: spec doesn't mandate one. Solver should fix and report it for reproducibility.
- **Eval script not provided.** Solver cannot self-test the scorer; they should implement MAP@3 themselves (`sum_{k=1..3} I(pred_k == y) / k` per row, mean across rows) and hit@3 (`I(y in pred[:3])`), report both, and label which one they optimized.
- **Naive baseline floor**: predicting the global top-3 (`West_Nile_fever Japanese_encephalitis Tungiasis`) for every row scores hit@3 ≈ 0.335 just from the class prior. Any submitted model that doesn't clear this is broken.

## 5. Spirit-of-spec checklist

- [ ] Predict `prognosis` for every row in `test.csv` (142 rows, ids 2..703). Ignore `sample_submission.csv`'s ids entirely.
- [ ] Output exactly the columns `id,prognosis`; `prognosis` is a single space-separated string of 3 distinct label tokens with underscores preserved (no spaces inside labels).
- [ ] Order the 3 labels by descending model confidence (defensible under both stated interpretations of MPA@3).
- [ ] Use only the 11 labels observed in `train.prognosis`; closed-set assumption.
- [ ] Train with stratified k-fold CV (k=5 reasonable for n=565, 11 classes); report mean ± std of hit@3 AND MAP@3.
- [ ] Beat the prior-only baseline (`hit@3 ≈ 0.335`) by a clear margin; if not, debug before submitting.
- [ ] Document any use of external data (default: none); fix and report a random seed.
