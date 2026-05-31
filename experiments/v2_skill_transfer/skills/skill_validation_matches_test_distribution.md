# Skill: Choose a validation protocol that mirrors the actual test distribution

## When this applies
Anywhere you compute a CV score and report it as your headline. Especially when the shipped split is non-obvious (random re-split, group-aware split, temporal split, or the prompt's description of the split disagrees with the file's reality).

## What to do
- Before picking a CV scheme, **inspect the train/test split empirically**:
  - Compare ID ranges, date ranges, day-of-month, group-id distributions.
  - Check `set(train[col]) ∩ set(test[col])` for any plausible group column.
  - Don't rely on the prompt's description of the split — it may be a lie or stale (saw on bike-sharing: prompt claimed days 20+, files showed days 1-19).
- Pick CV that mirrors the *actual* split distribution:
  - Random re-split of one original dataset → `StratifiedKFold` (or random `KFold` for regression) is correct, and group-aware CV is appropriate only if there's group leakage.
  - Group leakage present → `GroupKFold` / `StratifiedGroupKFold`.
  - Temporal split (true held-out future) → `TimeSeriesSplit` or held-out latest-N.
  - Small dataset (<2000 rows) → ≥5 folds, repeat if cheap, always report mean ± std.
- **Do not over-defensively pick the strictest possible CV** if the shipped test set is actually random — a pessimistic validation can push you toward a worse model (saw on bike-sharing: late-days holdout caused B to discard the better two-head approach).
- Cross-check: run *both* a per-row stratified split and a group/time-aware split, report both. The gap between them quantifies the leakage you should worry about.
- Report `mean ± std` over folds, not a single number. With small N the std is the load-bearing piece.

## Why
CV scores drive model selection. A CV protocol that doesn't match the test distribution will systematically prefer the wrong model — either by being too optimistic (random CV on group-leaked data) or too pessimistic (strict temporal CV on a randomly-shipped test set). Both errors propagate to the final submission.

## Code snippet
```python
# Step 1: empirically diagnose the split before picking CV
print("train date range:", train["date"].min(), train["date"].max())
print("test  date range:", test["date"].min(),  test["date"].max())
print("train ∩ test on group col:", len(set(train["group"]) & set(test["group"])))

# Step 2: run BOTH protocols when in doubt
from sklearn.model_selection import StratifiedKFold, GroupKFold

scores_per_row = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    scores_per_row.append(eval_fold(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]))

scores_group = []
for tr, va in GroupKFold(5).split(X, y, groups):
    scores_group.append(eval_fold(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]))

print(f"per-row CV: {np.mean(scores_per_row):.4f} ± {np.std(scores_per_row):.4f}")
print(f"group CV:   {np.mean(scores_group):.4f} ± {np.std(scores_group):.4f}")
# The gap = leakage magnitude. Pick the CV that matches test reality.
```

## Cross-task evidence
- Saw on: spaceship-titanic (per-row CV was +1.1pp optimistic vs group CV; A and C used per-row and over-reported), bike-sharing-demand (B's strict late-days CV mis-calibrated to the actually-random test split; C's looser time-based CV won), playground-series-s3e3 (small N=1341; stratified k-fold with std required, not single holdout), playground-series-s3e22 (N=988, 3 classes; 5-fold stratified mandatory), titanic (small N; reported CV5).
