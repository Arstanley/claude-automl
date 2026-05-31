# Ambiguity Audit — Titanic Passenger Data Modeling Task

Audit of `ambig_prompt.txt`. The spec leaves enough wiggle room that a careless solver can silently commit to wrong choices and still produce a confident-looking submission. This document lists those interpretive gaps so the solver must defend every choice.

## 0. Quick data reality check (so we can be specific, not generic)

- `train.csv`: 712 rows, 12 columns (spec says 713 — off by one).
- `test.csv`: 179 rows, 11 columns (spec says 180 — off by one).
- Together: 712 + 179 = 891 rows; combined `PassengerId` covers 1..891 with no gaps and no overlap. This is the canonical Kaggle `train.csv` (891 rows) randomly re-split.
- The only column in `train` not in `test` is `Survived` (int64, values {0,1}, 444 zeros / 268 ones — ~37.6% positive).
- `gender_submission.csv`: 418 rows, `PassengerId` range **892..1309** — i.e. the original Kaggle `test.csv` IDs, which **do not appear in this experiment's `test.csv` at all** (intersection = 0).
- All other categorical columns are populated in both splits. Missingness: `Age` ~20% (both), `Cabin` ~75%+ (both), `Embarked` 2 rows in train only.

## 1. Target-column ambiguity

The spec uses a placeholder `<target>` in the submission template ("Replace `<target>` with the column you are predicting") and never explicitly says "predict `Survived`". A lazy reader will default to `Survived` because that's the cultural prior for Titanic — but the spec invites them to "audit" rather than assume. Plausible candidates:

| Candidate | Reasoning for picking it | Reasoning against |
|---|---|---|
| **`Survived`** | Only column absent from `test.csv`. Standard Titanic task. Binary, matches "classification accuracy" eval. `gender_submission.csv` example uses this column name. | None on data grounds — this is the structurally indicated choice. |
| `Pclass` | 3-class multi-class; "ticket class" is also a classification target. | Present in `test.csv` — predicting it would be trivially leaky / nonsensical. |
| `Sex` | Binary. | Present in `test.csv`. Same leak issue. |
| `Embarked` | 3-class. | Present in `test.csv`. Same leak issue. Also has missing values even in train. |
| `Age` bucketed | Could be reframed as classification. | Spec doesn't mention bucketing; would be a stretch invented by solver to inflate options. |
| `Cabin` deck letter | Categorical, mostly missing. | 75% missing in train — not a credible target; not the structurally absent column. |

**What the data structure suggests:** `Survived`. It is the *only* column missing from test, the example submission file is literally named `gender_submission.csv` and has a `Survived` column, and the train label is already encoded {0,1} which lines up with "classification accuracy".

**What a lazy reader silently commits to:** also `Survived` — but for the *wrong reason* ("everyone knows the Titanic task"), without verifying that (a) the column is actually missing from test, (b) the example submission uses that column, and (c) the spec's `<target>` placeholder is consistent with that choice. The solver must defend the inference, not just the conclusion.

## 2. Other interpretive gaps

### 2a. Submission row count and PassengerId domain mismatch
- `test.csv` PassengerIds are **6..890** (179 rows, a random subset of 1..891).
- `gender_submission.csv` PassengerIds are **892..1309** (418 rows). These are the IDs from the *original* Kaggle holdout.
- **The example submission shares zero PassengerIds with this experiment's test set.** A solver that copies the example's row layout (rows 892..1309) will submit predictions for passengers they have no data for, and *no* predictions for the passengers actually in `test.csv`.
- The spec's submission template even shows `892,<value>` as the first row — actively misleading. The correct PassengerIds to predict are in `test.csv`, not the example file.

### 2b. Row count off-by-one in the spec
- Spec: "713 rows" train, "180 rows" test. Actual: 712 and 179. Not a modeling issue but a tell that the spec wasn't regenerated against the data; raises the prior that other details (column list, target name) may also be stale.

### 2c. Submission column header naming
- Spec says header should be `PassengerId,<target>`. If the target is `Survived`, the literal header should be `PassengerId,Survived` (capital S, no quotes, no trailing whitespace).
- A solver could ship `PassengerId,prediction`, `PassengerId,label`, `PassengerId,survived` (lowercase), or even `id,Survived`. Any of these is defensible against the literal `<target>` placeholder, but the example `gender_submission.csv` settles it: `PassengerId,Survived`.

### 2d. Prediction value type
- `gender_submission.csv` uses integer `0`/`1`. Train `Survived` is int64.
- Spec only says "classification accuracy" and "value" — does not forbid probabilities or strings ("yes"/"no", `True`/`False`). A solver outputting floats (`0.0`/`1.0`) or probabilities (`0.73`) could grade as 0% accuracy under exact-match scoring. The defensible choice is integer `0`/`1`.

### 2e. Evaluation = "classification accuracy"
- No mention of class-balance reweighting, F1, AUC, or any robustness check. Train is ~62/38 (majority = died). A solver could trivially game accuracy by predicting all-zero (~62% baseline) or by the gender heuristic (~75–80% historically). Neither is forbidden, but both would clearly violate the spirit of "model the passenger records".
- No information about test-set class balance. Solver should not assume parity with train.

### 2f. Data quirks the solver should not silently paper over
- `Age` is ~20% missing in both splits, and the spec notes "estimated ages are xx.5" — i.e. the `.5` decimal carries information (estimation flag), not biological precision. Bucketing or rounding away the `.5` destroys signal.
- `Cabin` is 75%+ missing — usable only via "is present" or deck-letter features, not raw string.
- `Embarked` has 2 missing values in train, 0 in test. A pipeline that imputes from train statistics still has to handle the 2 NaNs.
- `Ticket` and `Name` are high-cardinality strings; `Name` contains an extractable title (Mr./Mrs./Miss./Master.) that is a known strong signal — using it is fine, but a solver claiming "no leakage" while extracting titles should note that titles partially encode `Sex` and `Age`.
- `Fare` has no missing values in this split (canonical Kaggle test has 1 missing `Fare`; this random re-split avoids it). The solver should not hard-code "fare can be NaN in test" without checking.
- Because train and test are a random split of the original 891-row Kaggle train, **family members may be split across train and test** (same `Ticket`, same surname). A solver who builds family-survival features from train and applies them to test is technically leaking — defensible only if they declare it.

### 2g. The `gender_submission.csv` honeytrap
- It is named `gender_submission` and predicts "all females survive, all males die" — the trivial Kaggle baseline. A solver who treats it as a *target file to overwrite* (not just a format example) will: (i) submit 418 rows for the wrong PassengerIds, and (ii) potentially be anchored to the gender-only baseline. The spec calls it "an example submission file format" — format only, not row set, not values.

## 3. Spirit-of-spec checklist (defend every choice against these)

The solver must be able to answer "yes" to every bullet, in writing, before declaring done:

- [ ] **Target verified by structure, not folklore.** "I picked `Survived` because it is the *only* column in `train.csv` absent from `test.csv`, the example submission file's second column is `Survived`, and its dtype/cardinality matches a classification-accuracy metric." Not "because Titanic = survival."
- [ ] **Submission PassengerIds come from `test.csv`, not from `gender_submission.csv`.** The submission has exactly 179 rows (= `len(test)`), PassengerIds in 6..890, in the same order as `test.csv` (or sorted — but explicitly chosen, not accidental).
- [ ] **Submission header is literally `PassengerId,Survived`** and values are integers `0`/`1` (not floats, not booleans, not probabilities, not strings).
- [ ] **No feature in the model is derived from a column that is also the target, and no test-row label was used to fit.** In particular, family-survival / ticket-group features that pool train+test rows are flagged as a known leakage risk because the random split co-locates relatives.
- [ ] **Missingness is handled, not ignored.** `Age` (~20%), `Cabin` (~75%), `Embarked` (2 rows) each have an explicit policy. The `xx.5` "estimated age" flag is either preserved or its loss is justified.
- [ ] **Accuracy is reported with a baseline.** "Majority class predicts ~62%; gender-only predicts ~78%; my model predicts X%." A model that fails to beat the gender baseline has not earned its complexity.
- [ ] **The 712/179 row count was verified at load time** (and the spec's 713/180 discrepancy was noticed). If the solver's report quotes 713/180 verbatim, they never looked at the data.

---

End of audit. The solver's job is to make the right calls *and explain them*; this document exists so they can't claim the spec was unambiguous.
