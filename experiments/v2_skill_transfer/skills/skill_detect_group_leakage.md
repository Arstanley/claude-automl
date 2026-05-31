# Skill: Detect group/ID leakage and use group-aware CV

## When this applies
Any task where a non-target column links rows that share an underlying entity (a passenger group, household, patient, hospital, ticket, family, session, store, time-window). The risk is highest when train and test were produced by re-splitting a single original dataset.

## What to do
- For every ID-like or grouping column, compute the train/test overlap of unique values:
  - `set(train[col].unique()) & set(test[col].unique())` — overlap > 0 is a flag.
  - Fraction of test rows whose group also appears in train — anything >5% is material.
- If overlap exists and is large: use `GroupKFold` (or `StratifiedGroupKFold`) on that column for cross-validation, not random `KFold` or `StratifiedKFold`. Per-row CV will be optimistic by the leakage rate.
- Even if you choose not to engineer group-leakage features (e.g., "groupmate mean label"), the CV protocol still must split by group — otherwise reported metrics are inflated by between-fold leakage.
- If you do build groupmate-derived features, declare it in writing, justify under group-aware CV, and report the score without the feature for comparison.
- Watch for hidden group structure encoded inside ID strings (e.g. `gggg_pp` in spaceship-titanic). Parse and check.
- Time-based tasks: the analogous group is "same day / hour / week"; use a time-aware split (`TimeSeriesSplit` or held-out latest-N) instead of random.

## Why
Random CV on a leakage-heavy split silently inflates the headline by the order of the leakage rate (saw +1.1pp on spaceship-titanic). The model isn't actually generalizing — it's memorizing the group. Group-aware CV gives an honest signal of test-set performance and pushes feature engineering toward genuinely transferable signal.

## Code snippet
```python
from sklearn.model_selection import GroupKFold, StratifiedKFold

# 1. Probe for group leakage
for col in candidate_group_cols:  # e.g. ['hospital_number','PassengerGroup','user_id']
    tr_uniq, te_uniq = set(train[col].unique()), set(test[col].unique())
    overlap = tr_uniq & te_uniq
    leaked_rows = test[col].isin(overlap).mean()
    print(f"{col}: |overlap|={len(overlap)} / |test unique|={len(te_uniq)} ;"
          f" frac test rows with leaked group={leaked_rows:.1%}")

# 2. If overlap is material, build groups and use GroupKFold
groups = train["group_col"].values            # parse from ID if needed
cv     = GroupKFold(n_splits=5)
scores = []
for fold, (tr, va) in enumerate(cv.split(train, y, groups)):
    model.fit(X.iloc[tr], y.iloc[tr])
    scores.append(score_fn(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
print(f"GroupKFold {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

## Cross-task evidence
- Saw on: spaceship-titanic (40.8% test rows share `gggg` group with train; per-row CV was +1.1pp optimistic), playground-series-s3e22 (`hospital_number` 90% overlap), bike-sharing-demand (same-day temporal leakage — random split within days 1-19), titanic (random re-split puts family members across train/test via shared `Ticket`/surname).
