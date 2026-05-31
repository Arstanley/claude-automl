# Ambiguity Audit — Playground S3E22 (Horse Survival)

Auditor role: Challenger. Goal is to enumerate the interpretive gaps the
under-specified prompt leaves open, not to solve the task. Every claim below is
grounded in `data/train.csv` (988 rows, 29 cols), `data/test.csv` (247 rows, 28
cols), and `data/sample_submission.csv` (824 rows, 2 cols).

---

## 1. Target-column ambiguity

The prompt says "predict the target prediction column" but never names it. The
column list is prefixed "Examples of columns include …" — explicitly partial.
Candidates the solver must consider and rule in/out:

| Candidate          | Type in train             | In test?       | Verdict                                   |
| ------------------ | ------------------------- | -------------- | ----------------------------------------- |
| `outcome`          | categorical, 3 classes    | **MISSING**    | **The target.** Only column dropped from test. |
| `surgery`          | binary {yes,no}           | present        | Distractor — fully observed in test.      |
| `surgical_lesion`  | binary {yes,no}           | present        | Distractor — fully observed in test.      |
| `cp_data`          | binary {yes,no}           | present        | Distractor — fully observed in test.      |
| `hospital_number`  | high-card int (241 uniq)  | present        | ID-like, not a target.                    |
| `lesion_1`         | 56-value int code         | present        | Distractor; categorical-coded but in test.|

The decisive signal — the only one the solver should rely on — is the
**schema diff**: `set(train.columns) - set(test.columns) == {'outcome'}`. Any
solver who picks a different target is gaming or guessing.

Corroborating signals (do NOT use as primary justification, since they could be
absent in a more adversarial prompt):

- `sample_submission.csv` header is literally `id,outcome` and the placeholder
  value column is filled with the string `"lived"` — one of the three outcome
  categories.
- `outcome` is the only column in train whose values
  (`lived`/`died`/`euthanized`) match the F1-micro-implies-classification
  requirement *and* are absent from test.

**Defend-against question for the solver:** "What is the column-set difference
between train and test, and which candidate in that diff is categorical?" If
their answer cites the sample submission or the prompt text rather than the
schema diff, they have skipped the load-bearing check.

### Target distribution (for sanity, not modeling guidance)

```
lived       453   (45.8%)
died        338   (34.2%)
euthanized  197   (19.9%)
```

Three classes, moderate imbalance. Micro-F1 on a multi-class problem reduces to
accuracy when each row has exactly one predicted label — the solver should
state this explicitly so they don't waste effort on per-class threshold tuning
that micro-F1 ignores.

---

## 2. F1-micro forces categorical target

The metric is "micro-averaged F1-Score". This rules out:

- Regression targets (`rectal_temp`, `pulse`, `respiratory_rate`,
  `packed_cell_volume`, `total_protein`, `abdomo_protein`,
  `nasogastric_reflux_ph`) — F1 is undefined for continuous outputs without a
  discretization the prompt does not provide.
- High-cardinality ID-like columns (`hospital_number`, `lesion_1`) — micro-F1
  on 50+ classes with <1000 training rows is technically defined but spec-wise
  nonsensical.

The metric + the schema diff jointly pin the target to `outcome`. A solver who
proposes anything else is generating, not deducing.

---

## 3. Other interpretive gaps

### 3a. Sample-submission row-count mismatch (likely trap)

- `test.csv` has **247** rows.
- `sample_submission.csv` has **824** rows.
- `test.csv` ids range `[10, 1230]`, 247 unique; `train.csv` ids range
  `[0, 1234]`, 988 unique; **zero overlap** between train and test ids.

The sample submission is stale or copied from the original Kaggle competition
(which had a larger test set). The solver must (a) notice this, (b) submit
predictions only for ids present in `test.csv`, and (c) not blindly join on the
sample submission's id list. A naive `submission = sample_submission.copy();
submission['outcome'] = pred` pipeline will silently produce wrong output.

### 3b. `hospital_number` is an ID, not a feature — but with leakage potential

- 241 unique values in train, 138 in test.
- **124 hospital_numbers appear in BOTH train and test** (out of 138 test
  uniques — 90% overlap).

This is leak-shaped: a horse seen in both splits via the same hospital can let
a tree model encode patient-level priors. The solver should either drop the
column, target-encode it with out-of-fold smoothing, or explicitly defend
treating it as a feature. Silent label-encoding is the worst option.

### 3c. `lesion_1` / `lesion_2` / `lesion_3` are coded categoricals, not counts

- `lesion_1`: int dtype but **56 unique codes** in train, 38 in test. Top codes
  (`2208`, `3205`, `2209`, `2205`, `3111`) look like 4-digit anatomical/severity
  composites, not magnitudes. Treating as numeric is wrong.
- `lesion_2`: 984/988 = 99.6% zero. Effectively a rare-indicator.
- `lesion_3`: 987/988 = 99.9% zero. Effectively a single-row outlier flag.

Solver must defend their encoding choice for `lesion_1` (one-hot? target
encode? digit-split into body-region/severity?) and justify keeping
`lesion_2`/`lesion_3` at all given near-zero variance.

### 3d. Missingness is real and informative

Top missing columns in train (% NaN):

```
abdomen                17.6%
rectal_exam_feces      15.3%
nasogastric_tube        6.2%
peripheral_pulse        4.9%
abdomo_appearance       3.8%
pain                    3.2%
```

Test missingness mirrors train within ~2pp on every column — so missingness is
**not** a train/test distribution shift. But "field not recorded" in veterinary
records is plausibly correlated with case severity (a dying horse may not get a
full exam). Solver should consider missing-indicator features rather than
silent median/mode imputation, and defend the choice.

### 3e. Categorical columns with many ordered levels

Columns like `pain`, `peripheral_pulse`, `mucous_membrane`,
`capillary_refill_time`, `abdominal_distention` are clinically **ordinal**
(`mild` < `moderate` < `severe`-style) but stored as strings. The solver must
decide between ordinal encoding (requires domain ordering they were not
given) vs. one-hot (loses ordering). Either is defensible; silently
label-encoding alphabetically is not.

### 3f. Synthetic-data caveat from the prompt

The prompt explicitly says the data is generated by a deep model from the
original Horse Survival dataset. Implication: feature correlations may be
weaker or distorted vs. the real dataset, and any "domain knowledge" lifted
from the original UCI Horse Colic dataset may not transfer cleanly. Solver
should not import hand-crafted feature rules from the literature without
validating on this synthetic distribution.

### 3g. Train size is tiny (988 rows, 3 classes)

With ~197 `euthanized` rows, a single 80/20 split produces ~40
minority-class validation examples. CV variance will be large. Solver must
report cross-validated micro-F1 with at least 5 folds and standard
deviation, not a single hold-out score, or any reported number is noise.

---

## 4. Spirit-of-spec checklist (solver must defend each)

1. **Schema-diff justification.** Did you derive the target from
   `set(train.columns) - set(test.columns)` rather than from the prompt text
   or the sample submission file? Cite the column.
2. **Submission row count.** Does your final submission have exactly **247**
   rows matching `test.csv['id']`, not 824 from the stale sample submission?
3. **Categorical target preserved.** Are predictions emitted as the literal
   strings `lived` / `died` / `euthanized` (not 0/1/2, not one-hot)? Micro-F1
   compares string equality at the row level.
4. **`hospital_number` handled deliberately.** Did you drop it, target-encode
   with OOF folds, or justify keeping it raw given the 90% train/test
   overlap? Silent label-encoding is disqualifying.
5. **`lesion_1` treated as categorical.** Did you avoid feeding the 4-digit
   codes as numeric magnitude into a linear/distance model? Did you defend
   keeping or dropping `lesion_2`/`lesion_3` given their >99% zero rate?
6. **Missingness modeled, not silently filled.** Did you add missing-indicator
   features for the high-missing columns (`abdomen`, `rectal_exam_feces`,
   `nasogastric_tube`) or justify why simple imputation is sufficient?
7. **Reported score is cross-validated, not single-split.** With n=988 and a
   3-class target, report mean ± std micro-F1 across ≥5 stratified folds. A
   single hold-out number is not defensible.

A solver that cannot answer all seven without hand-waving is gaming the spec,
not solving it.
