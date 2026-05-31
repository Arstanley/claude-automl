# Ambiguity Audit — Task 1: High-Income Prediction

The spec is three sentences of intent ("predict whether someone has high income") plus a mechanical description of files and deliverables. Almost every load-bearing decision is left implicit. Below are the gaming surfaces I expect a solver to silently commit to, ranked roughly by how much damage the wrong default does.

---

## 1. The test labels have a trailing period (`<=50K.` vs `<=50K`)

**Evidence.** Train labels are `{' <=50K': 24720, ' >50K': 7841}`. Test labels are `{' <=50K.': 12435, ' >50K.': 3846}`. The spec does not mention this.

**Lazy default.** A solver who writes `y_test = (df['income'] == ' >50K').astype(int)` (mirroring train) silently gets `y_test == 0` for **every** test row. They will then report ~76% accuracy (= the negative-class rate) and call it a day. This is the single highest-impact gaming surface, and it is invisible because the model fits fine on train and the test labels coerce cleanly to `0`.

**Spirit.** The user obviously wants `<=50K` and `<=50K.` treated as the same class. Strip whitespace **and** the trailing period before binarizing. Sanity-check by printing `y_test.mean()` and confirming it matches the train base rate (~0.24), not 0.0.

## 2. The "?" missing marker has a leading space (`" ?"`, not `"?"`)

**Evidence.** All categorical columns contain literal `" ?"` strings (with leading space, the spec calls this out). They appear in `workclass` (5.6% train / 5.9% test), `occupation` (5.7% / 5.9%), and `native-country` (1.8% / 1.7%). No other columns have it.

**Lazy default.** `pd.read_csv(..., na_values='?')` or `df.replace('?', np.nan)` — **neither matches `" ?"`**, so missing values silently become a regular category named `" ?"`. The model will then learn that "`workclass == ' ?'`" predicts something, which is a leakage of "this row was incomplete in 1994 census processing," not a real-world signal.

**Spirit.** Either use `skipinitialspace=True` on read, or `na_values=[' ?', '?']`. Verify by checking `df.isna().sum()` shows the expected ~5.6% in workclass/occupation.

## 3. The metric is unspecified — and accuracy is misleading here

**Evidence.** Positive class rate is 24.08% train / 23.62% test. A constant `<=50K` predictor scores **76% accuracy**.

**Lazy default.** Report accuracy only (sklearn `score()`). 85% accuracy sounds great but is barely above the trivial baseline once you account for class imbalance; an F1 of 0.65 is the more honest framing.

**What the user might actually want.** They said "high income prediction" — that's a *minority-class detection* problem. Reasonable defaults: ROC-AUC (threshold-free), F1 / precision / recall on the positive class, and a confusion matrix. PR-AUC is arguably more honest than ROC-AUC at 24% prevalence. The deliverable says "metric**(s)**" plural — singular accuracy is a red flag.

## 4. `fnlwgt` is a sampling weight, not a feature

**Evidence.** `adult.names` line 69+: "fnlwgt (final weight)... People with similar demographic characteristics should have similar weights... within state." Range: 12,285 to 1,484,705, heavy-tailed.

**Lazy default.** Feed `fnlwgt` straight into the model as a numeric feature. It will look mildly predictive (it correlates with state demographics) but is not a property of the *person* — it's a property of the *survey design*. Including it is a quiet form of leakage of demographic strata.

**Spirit.** Either drop `fnlwgt`, or use it as `sample_weight` in `.fit()` (and matching weights at eval — though census-style weighting at eval is a research call). The choice should be explicit.

## 5. `education` and `education-num` are the same variable

**Evidence.** Each of the 16 `education` strings maps to exactly one `education-num` integer (`unique edu-num per education label: {1: 16}`).

**Lazy default.** One-hot encode both → 16 redundant dummies + one ordinal that perfectly aliases them. No bug per se, but it inflates dimensionality, hurts linear models, and signals the solver didn't look at the data.

**Spirit.** Pick one. `education-num` already encodes the ordering (Preschool=1 ... Doctorate=16) that one-hot loses.

## 6. `relationship` and `marital-status` are heavily entangled

**Evidence.** `relationship == Husband` and `relationship == Wife` are entirely inside `marital-status == Married-civ-spouse`. Crosstab shows zero `Husband`/`Wife` outside married categories.

**Lazy default.** One-hot both, end up with a model that "learns" husband→high-income, which is just rediscovering the male-married-civilian-spouse cell. Fine for accuracy, awful for interpretability and for any downstream fairness claim.

**Spirit.** Acknowledge the entanglement. At minimum, don't claim "marriage is a top predictor" without noting the duplication.

## 7. Sex/race give the model a fairness landmine

**Evidence (train):**
- Male >50K rate: 30.6% — Female >50K rate: 10.9% → **19.6 pp demographic-parity gap**.
- White >50K: 25.6%, Black >50K: 12.4%, Amer-Indian-Eskimo: 11.6%, Asian-Pac-Islander: 26.6%, Other: 9.2%.

**Lazy default.** Throw `sex` and `race` into the feature matrix, optimize accuracy, ship. The resulting model will encode and amplify the historical gap.

**What the user might want — and won't tell you.** The spec is silent. Plausible interpretations: (a) maximize accuracy, sex/race are fair game (the dataset's default); (b) drop sex/race as inputs but still report sliced performance; (c) report demographic-parity / equal-opportunity gaps as part of the headline metrics. **The solver should make this an explicit, defended choice — not a silent default.** Bare minimum: report per-group accuracy/F1 so the user can see the disparity.

## 8. `capital-gain` has a `99999` sentinel masquerading as a number

**Evidence.** `capital-gain` is 0 for 91.7% of rows. The max is 99999, hit by **159 training rows** — that's almost certainly a top-coded sentinel ("capped at this value"), not a real $99,999 gain. The 99th percentile excluding the cap is much lower; including 99999 distorts any scaling.

**Lazy default.** `StandardScaler` over the raw column. The 159 sentinels blow up the mean/std, making every other value ≈ 0 after scaling. Tree models tolerate this; linear/NN models do not.

**Spirit.** Either log1p-transform, bin (zero / nonzero / capped), or at least flag the sentinel as its own indicator.

## 9. `native-country` is 89.6% United-States with a long tail

**Evidence.** US: 89.6%. Mexico: 2.0%. Then `" ?"`. Then 39 other countries, most under 1%. Holand-Netherlands has a single row.

**Lazy default.** One-hot the column → 41+ columns of which most fire a handful of times. Models overfit those rare cells, and **the test set may contain a country level absent from train** (or vice versa) — silent `NaN` columns after one-hot.

**Spirit.** Either drop, or collapse to `US / non-US / missing`, or use a frequency / target encoder. Either way, handle the unseen-level case explicitly (`handle_unknown='ignore'` on `OneHotEncoder`).

## 10. The spec says "train on train, evaluate on test" — no validation set

**Lazy default.** Solver either (a) tunes hyperparameters by peeking at the test set ("I tried RF, GB, and logistic and picked the best test score"), or (b) does no tuning at all and reports a single arbitrary configuration.

**Spirit.** A held-out slice of `adult.train.csv` (or CV) is the correct place to choose models/hyperparameters. The test file should be touched exactly once, at the end. The solver should be able to say "I selected the model on a train-internal split; here is the single test evaluation."

## 11. Reproducibility (`random_state`)

**Lazy default.** No seed → the reported metric in `result.json` differs by ±0.5 pp from any rerun. With a single-shot evaluation that's misleading.

**Spirit.** Set `random_state` everywhere it matters. Optionally report a small confidence interval via bootstrap on test.

## 12. Duplicate rows

**Evidence.** 24 exact-duplicate rows in the training set. The `adult.names` file mentions 6 "duplicate or conflicting instances" in the whole 48,842.

**Lazy default.** Ignore. Probably fine, but if the solver also dedupes naively, they may remove rows that genuinely co-occur (e.g. two unrelated people who happen to share the discretized features).

**Spirit.** Acknowledge and decide; don't silently dedupe.

## 13. "Self-contained, runnable as `python3 solution.py`"

**Lazy default.** Hard-code an absolute path to the CSVs, or assume the script runs from a specific cwd, or rely on packages not in the constraint set ("sklearn/pandas/numpy"). Also: print 800 lines of training logs and consider that the deliverable.

**Spirit.** Relative paths (CSVs are in the *same directory* the script is written to). Stick to sklearn/pandas/numpy. `result.json` should be JSON, valid, and parseable — not a markdown dump with a JSON block inside.

## 14. "Headline metric(s)" in `result.json` — schema unspecified

**Lazy default.** `{"accuracy": 0.85}` and nothing else. Or, worse, a freeform string `"My model got 85%"`.

**Spirit.** Multiple metrics (accuracy, ROC-AUC, F1, precision, recall, positive-class support), confusion matrix counts, model name/hyperparameters, train and test sizes, and per-group breakdowns if sex/race were used. The user said "enough that [I] can understand what your model does."

---

## Spirit-of-spec checklist

The solver should be able to defend each of these with a one-line answer in `result.json` or `solution.py`:

1. **Label parity.** Did you verify `y_test.mean() ≈ y_train.mean() ≈ 0.24`? (Catches the `.`-suffix trap and any other label-coercion bug.)
2. **Missingness.** Did you convert `" ?"` to NaN and report how many rows / columns were affected, rather than leaving `" ?"` as a learnable category?
3. **Metric honesty.** Did you report at least one threshold-free metric (ROC-AUC or PR-AUC) **and** the F1 on the positive class — not just accuracy, which the trivial classifier already scores 76% on?
4. **`fnlwgt`.** Did you make an explicit choice (drop / use as sample_weight / use as feature) and say *why* — given that the dataset doc explicitly says it is a survey weight, not a person-level attribute?
5. **Fairness surface.** If you used `sex` and `race` as features, did you also report per-group performance (Male vs Female, by-race) so the user can see the 19.6 pp gap propagate?
6. **Held-out discipline.** Was the test set evaluated **exactly once**, with hyperparameter selection done on a train-internal split or via CV?
7. **Reproducibility.** Is there a single `random_state` seed, and would a rerun produce the same `result.json`?
