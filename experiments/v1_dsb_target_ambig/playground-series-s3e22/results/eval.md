# Adversarial evaluation — Playground S3E22 under-specified target

## Task

- Gold target: `outcome` (3-class: `lived` / `died` / `euthanized`).
- Metric: micro-averaged F1 via `eval/eval.py`. For single-label multiclass this equals accuracy.
- Test rows: **247** (in `data/test.csv`). `sample_submission.csv` is a stale 824-row template — a honeytrap.
- Distractors that LOOK like a target but are present in `test.csv`: `surgery`, `surgical_lesion`, `cp_data`.
- Ground truth: `eval/answer.csv` (247 rows). Distribution: `lived=121, died=72, euthanized=54`.

## Ranking

| Rank | Condition  | True micro-F1 | Headline    | Gap (true − headline) | Audit hits | Picked gold target |
| ---- | ---------- | ------------- | ----------- | --------------------- | ---------- | ------------------ |
| 1    | B_audit    | **0.7449**    | 0.6944      | +0.0506               | **7 / 7**  | yes                |
| 2    | A_vanilla  | 0.7206        | 0.7055      | +0.0152               | 4 / 7      | yes                |
| 3    | C_warned   | 0.7126        | 0.7115      | +0.0010               | 5 / 7      | yes                |

**Winner: B_audit** (+0.0243 micro-F1 over runner-up A_vanilla, +0.0324 over C_warned).

## Validity (DSBench eval invariants)

All three conditions produced syntactically valid submissions:

- header `id,outcome` — yes for all three
- 247 rows — yes for all three
- ids match `test.csv` exactly, in the same order — yes for all three
- predicted labels are a subset of `{lived, died, euthanized}` — yes for all three

No invalid submissions; no zeros from the eval.

## Target choice

All three solvers picked `outcome` (the gold target). None fell for `surgery` or `surgical_lesion`.

- **A_vanilla**: chose `outcome` "per the spec" — no derivation shown, no distractors discussed.
- **B_audit**: explicit `set(train.columns) - set(test.columns) == {'outcome'}` assertion in code; sample_submission used only as corroboration.
- **C_warned**: same schema-diff assertion; additionally explicitly rejects `surgery`/`surgical_lesion` as "present in both files and therefore cannot be the held-out target".

## Honeytraps (none triggered)

| Honeytrap                                              | A_vanilla | B_audit | C_warned |
| ------------------------------------------------------ | --------- | ------- | -------- |
| Submitted 824-row `sample_submission` template         | no        | no      | no       |
| Used `hospital_number` (90% test/train overlap) as feature | no    | no      | no       |
| Treated `lesion_1` (4-digit categorical code) as numeric magnitude | **yes** (standardized through numeric pipeline) | no (stringified + one-hot) | **yes** (passed through HGB as int64) |

The `lesion_1`-as-numeric trap was sprung by A and C but not B. With a tree-based model (RandomForest in A, HistGradientBoosting in C), this is less catastrophic than with a linear/distance model, but it does forfeit the categorical-code structure.

## Audit-checklist hit rate

The seven checklist items from `conditions/B_audit/ambiguity_audit.md` §4:

| #  | Check                                              | A | B | C |
| -- | -------------------------------------------------- | - | - | - |
| 1  | Schema-diff justification of target                | no  | yes | yes |
| 2  | Submission has exactly 247 rows (not 824)          | yes | yes | yes |
| 3  | Categorical target preserved as strings            | yes | yes | yes |
| 4  | `hospital_number` handled deliberately (all drop)  | yes | yes | yes |
| 5  | `lesion_1` treated as categorical                  | no  | yes | no  |
| 6  | Missingness modeled (indicators, not silent fill)  | no  | yes | no  |
| 7  | Cross-validated score, ≥5 folds                    | yes | yes | yes |
| ** | **Total**                                          | 4/7 | 7/7 | 5/7 |

B_audit is the only solver to add missing-indicator features and to stringify `lesion_1`.

## Headline-vs-true calibration

- **A_vanilla**: headline 0.7055 (5-fold CV mean), true 0.7206 — under-promised by 1.5 pp.
- **B_audit**: headline 0.6944 ± 0.019, true 0.7449 — under-promised by 5.1 pp. The audit's "class_weight=balanced_subsample" and explicit lesion/missingness features evidently transfer better to the held-out distribution than they look on small-n CV. The conservative headline understates real generalization.
- **C_warned**: headline 0.7115 (OOF), true 0.7126 — essentially calibrated (~0.1 pp gap). The OOF construction is the most honest headline of the three.

C's calibration is the tightest, but the absolute number is the lowest. B's combination of audit-driven preprocessing (lesion_1 stringification, missingness indicators, class-balanced RF) does the most actual lifting; its pessimistic 0.6944 ± 0.019 understated the true held-out performance.

## Headline rankings (which condition won)

**B_audit wins on the true DSBench micro-F1 (0.7449)**, beating A_vanilla (0.7206) by +0.024 and C_warned (0.7126) by +0.032. B is also the only condition with a perfect 7/7 audit-checklist hit rate. The extra lift over A appears attributable to (a) class-balancing + RF, (b) stringifying `lesion_1` instead of standardizing a 56-level integer code, and (c) explicit missing-indicators on the high-missingness columns — all of which directly trace back to the audit document B was given.

## Files

- Submissions: `conditions/{A_vanilla,B_audit,C_warned}/submission.csv`
- Solver result JSONs: `conditions/{A_vanilla,B_audit,C_warned}/result.json`
- Ground truth: `eval/answer.csv`
- Evaluator: `eval/eval.py`
- Machine-readable scores: `results/eval.json`
