# Eval — playground-series-s3e5 (wine quality, QWK, N=10)

All three solvers produced **valid** submissions: header `Id,quality`, 412 rows,
Ids match `test.csv` in original row order, integer predictions inside `[0,9]`.
No honeytraps tripped (no `sample_submission.csv` Ids, no floats, no out-of-range).

## True QWK (DSBench `eval/eval.py`)

| Condition | Test QWK | Reported headline | Gap | Audit hits |
|---|---:|---:|---:|---:|
| A_no_skills    | **0.46649** | 0.50390 | +0.037 | 6.5 / 7 |
| A_with_skills  | **0.48163** | 0.54013 | +0.059 | 7 / 7 |
| B_audit_round2 | **0.48107** | 0.56757 | +0.087 | 7 / 7 |

Constant predictor → 0.0; spec target ≥0.45 met by all three.

## v2 deltas

- **Transfer signal** `A_with - A_no` = **+0.0151** QWK. The skill library *does* help vs vanilla — the gain comes entirely from `skill_tune_dont_default` driving post-hoc monotone threshold tuning on OOF regression predictions, which the vanilla solver omitted.
- **Ceiling gap** `B_audit - A_with` = **-0.0006** QWK (essentially zero, B slightly *behind* A_with on true test QWK). Per-task audit gave B a richer CV report (ensemble + constant-sanity) and a higher *headline* (0.568 vs 0.540), but the extra ensemble machinery did not translate to test gain. Both saturated the same plateau.

## Approach notes

- **A_no_skills**: `RandomForestRegressor` (best of 6 model variants) with naive round+clip to [3,8]. No threshold tuning — the one missed audit item, and the source of the ~0.015 QWK gap.
- **A_with_skills**: HistGradientBoostingRegressor + 5-threshold tuning on OOF. Explicitly cites 8 of 10 skills applied; correctly skips 2 with reasoning (no groups, no NaN).
- **B_audit_round2**: GBR+RF+ET ensemble + 5-threshold tuning + constant-predictor sanity check + Id-feature exclusion. Most rigorous audit-checklist coverage, but ensemble didn't beat the single-model A_with_skills on test.

## Headline-vs-true gap

B has the largest optimism (+0.087): its CV headline of 0.568 overstates test by ~15%. A_with_skills also optimistic (+0.059). A_no_skills is the least optimistic — partly because it didn't push thresholds. None show leakage-scale gaps (>0.15).

## Bottom line

- Skill library delivers a small but real transfer gain (+0.015 QWK) by enforcing the ordinal-aware recipe.
- The per-task audit ceiling is at parity with skill-library performance on this task; the additional ensembling in B is a wash on test.
