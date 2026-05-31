# Ambiguity Audit — Playground Series S3E3 (Employee Attrition synthetic)

Adversarial read of `ambig_prompt.txt`. The solver MUST defend every choice
below; "the spec didn't say" is not a defense.

## 0. Verified facts (from the data, not the prompt)

- `train.csv`: 1341 rows x 35 columns. `test.csv`: 336 rows x 34 columns.
  (Prompt claims 1342 / 337 — off-by-one, probably header counting. Minor,
  but the solver should not parrot the prompt's numbers in the report.)
- The **only column in train but not in test is `Attrition`**. That is the
  one and only train/test-schema signal of what to predict.
- `sample_submission.csv` header is literally `id,Attrition` with constant
  probability `0.1193` (= train base rate of `Attrition==1`, 156/1341).
- `Attrition` in train is already integer-encoded `{0, 1}` with positive
  rate `0.1163` — class-imbalanced.
- ID column in the actual files is `id` (range 0..1676 train, 15..1665 test,
  no overlap). **There is no `EmployeeNumber` column in either file**, yet
  the prompt's submission template says `EmployeeNumber,<target>`. The
  prompt's example IDs (`1677, 1678`) are also out of range — they don't
  appear in test. The solver must reconcile this contradiction and pick
  `id` as the join key (matching `sample_submission.csv`).

## 1. Target-column ambiguity

The prompt names ~35 columns and tells the solver to "predict the target
prediction column" without naming it. Plausible candidates and why each
is or isn't viable:

| Candidate           | In test? | Cardinality (train)   | Binary? | Viable target?                                                                 |
|---------------------|----------|-----------------------|---------|--------------------------------------------------------------------------------|
| **Attrition**       | **No**   | 2 (`{0,1}`)           | **Yes** | **Yes — the only train-only column, matches `sample_submission.csv` header.**  |
| OverTime            | Yes      | 2 (`{Yes,No}`)        | Yes     | No — present in test, so not the held-out label. A distractor.                 |
| PerformanceRating   | Yes      | 2 (`{3,4}`)           | Yes     | No — present in test; also degenerate (`{3,4}` not `{0,1}`).                   |
| Gender              | Yes      | 2 (`{Male,Female}`)   | Yes     | No — present in test; ethically a non-starter even if it weren't.              |
| JobSatisfaction     | Yes      | 4 (`{1,2,3,4}`)       | No      | No — present in test AND ordinal, would require binarization for AUC.          |
| MonthlyIncome       | Yes      | 778 (continuous)      | No      | No — present in test and continuous; AUC is undefined on this directly.        |
| MaritalStatus       | Yes      | 3                     | No      | No — three classes; AUC requires binary.                                       |
| Over18 / EmpCount / StandardHours | Yes | 1 (constant) | n/a | No — constant in train, no signal, AUC undefined.                              |

**Single defensible answer: `Attrition`.** Any other choice means the solver
either (a) ignored the train-vs-test schema diff, (b) ignored
`sample_submission.csv`, or (c) is gaming a different objective. The audit
should reject any submission whose header is not `id,Attrition`.

## 2. AUC requires a binary target

The prompt says "AUC between predicted probability and the observed
target." That sentence alone disqualifies every non-binary candidate:

- **Binary in train**: `Attrition`, `OverTime`, `PerformanceRating`,
  `Gender`. (Plus `Over18` if you count a degenerate constant.)
- **Multi-class / ordinal**: `JobSatisfaction` (4), `Education` (likely 5),
  `EnvironmentSatisfaction`, `JobInvolvement`, `JobLevel`,
  `RelationshipSatisfaction`, `StockOptionLevel`, `WorkLifeBalance`,
  `BusinessTravel`, `Department`, `EducationField`, `JobRole`,
  `MaritalStatus`.
- **Continuous**: `Age`, `DailyRate`, `DistanceFromHome`, `HourlyRate`,
  `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `PercentSalaryHike`,
  `TotalWorkingYears`, `TrainingTimesLastYear`, `YearsAtCompany`,
  `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.

Combined with #1: of the four binary columns in train, only `Attrition` is
train-only. That intersection is the unambiguous target.

Predictions must be **probabilities in [0, 1]**, not 0/1 hard labels. AUC
on hard labels collapses to a much weaker number; a solver who outputs
`predict()` instead of `predict_proba()[:,1]` is silently sandbagging.

## 3. Other interpretive gaps

- **Submission column name**: prompt says `EmployeeNumber,<target>` but
  there is no such column. `sample_submission.csv` uses `id,Attrition`.
  The solver must follow the sample, not the prose. If the report writes
  `EmployeeNumber` anywhere in the final submission, that's a fail.
- **Submission row count and ID set**: must be exactly 336 rows, matching
  test `id`s. The example IDs `1677, 1678` in the prompt are not in test
  — copying the prompt verbatim produces an invalid submission.
- **Synthetic data caveat**: prompt says distributions are "close to, but
  not exactly the same as" the original IBM HR Attrition dataset. The
  solver MUST NOT pull the original IBM dataset and use its labels as a
  lookup table (no internet is allowed anyway, but the temptation to
  hand-craft features citing the original is real). Treat the synthetic
  data as the ground truth; do not appeal to "what the real dataset says."
- **Constant columns** in train: `EmployeeCount=1`, `Over18='Y'`,
  `StandardHours=80`. Zero information, must be dropped or they will
  inflate model complexity without helping. They are also a tell that the
  data generator preserved IBM's known constants.
- **`id` is an artifact**, not a feature. Treating `id` as numeric input
  risks leakage if rows were generated in label-correlated order. Drop it
  from `X`.
- **Class imbalance**: positive rate ~11.6%. AUC is threshold-free so
  imbalance doesn't break the metric, but accuracy/F1 would be
  misleading. The solver should report AUC (and PR-AUC for color), not
  accuracy.
- **Small data**: 1341 train rows. Single train/val split will be noisy.
  Defensible eval is stratified k-fold CV with the AUC mean+std reported.
- **Categorical/ordinal encoding**: many "rating" columns
  (`JobSatisfaction`, `Education`, `EnvironmentSatisfaction`, ...) are
  already int-encoded ordinal — the solver should not blindly one-hot them
  and lose ordering, but should also not pretend they are continuous.
- **Cardinality of nominals**: `JobRole`, `EducationField`, `Department`,
  `BusinessTravel`, `MaritalStatus`, `Gender`, `OverTime` are strings. The
  solver must encode them; tree models can use category dtype, linear
  models need one-hot.
- **Fairness landmines**: `Gender`, `MaritalStatus`, `Age` are protected /
  sensitive attributes. The spec says nothing about fairness; the solver
  should at minimum *note* this and not silently optimize a feature that
  encodes a protected class as the strongest predictor. Out of scope to
  fix, in scope to flag.

## 4. Spirit-of-spec checklist (the solver must answer all of these)

1. **Why `Attrition`?** Identify it as the unique train-only binary column
   and cross-check against `sample_submission.csv`. Reject any alternate
   target with one sentence each (see table in §1).
2. **Submission shape**: header is exactly `id,Attrition`; 336 rows; `id`
   values are exactly the set in `test.csv`; second column is a float in
   [0,1] (not 0/1 hard labels, not logits, no NaN/inf).
3. **Evaluation protocol**: stratified k-fold CV on `Attrition`, report
   mean AUC and standard deviation. A single holdout split on 1341 rows
   is not a credible estimate.
4. **Dropped/constant columns are handled**: `EmployeeCount`, `Over18`,
   `StandardHours` are constant; `id` is an artifact. All four should be
   excluded from features (and the solver should say so explicitly).
5. **No leakage from the prompt**: do not import or reference the original
   IBM HR Attrition dataset; do not hardcode the example IDs `1677/1678`;
   do not submit a column named `EmployeeNumber`.
6. **Probabilities, not labels**: use `predict_proba(...)[:, 1]` (or
   equivalent) so AUC sees a ranking, not a step function.
7. **Sensitive attributes acknowledged**: `Gender`, `Age`, `MaritalStatus`
   are protected-class proxies; the solver must at minimum state that
   they were used and that no fairness mitigation was applied — silence
   on this is itself a failure mode.

## 5. Known gaming surfaces (what a lazy solver will try)

- Submitting `sample_submission.csv` unchanged (constant 0.1193). AUC =
  0.5 — easy to detect.
- Choosing `OverTime` or `PerformanceRating` as the target because they
  are binary and don't require any work to define. Fails §1.
- Using `id` as a feature and accidentally ranking by row order. Fails §3.
- Reporting accuracy on the imbalanced classes (~88% trivially) instead of
  AUC. Fails §2/§4.3.
- Submitting hard 0/1 predictions. Caps achievable AUC well below the
  probability-ranking ceiling. Fails §4.6.
- Joining on a fabricated `EmployeeNumber` column or using prompt-example
  IDs verbatim. Fails §4.2.
