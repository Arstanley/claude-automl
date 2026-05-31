# Skill: Include class_weight in the grid for imbalanced binary classification

## Precondition
Apply if ALL hold (mechanically checkable on `train[target]`):
1. Binary classification — `train[target].nunique() == 2`.
2. Minority-class rate `< 0.20` (positive imbalance) — `min(train[target].value_counts(normalize=True)) < 0.20`.
3. `n_train >= 500`.
4. Headline metric in `{"roc_auc", "auc", "f1", "accuracy", "average_precision", "pr_auc", "logloss"}`.

```python
vc = train[target].value_counts(normalize=True)
preco = (
    train[target].nunique() == 2
    and float(vc.min()) < 0.20
    and len(train) >= 500
    and headline_metric.lower() in {"roc_auc","auc","f1","accuracy","average_precision","pr_auc","logloss","log_loss"}
)
```

## Action
Add `class_weight` (or model-equivalent) as an **extra grid axis**, not a free hyperparameter tuned in isolation:

- For sklearn `RandomForest*` / `LogisticRegression` / `GradientBoosting*` (with `sample_weight`): add `class_weight in {None, "balanced"}`.
- For `HistGradientBoostingClassifier`: it does not accept `class_weight`; instead pass `sample_weight = compute_sample_weight("balanced", y)` to `.fit` as one variant.

Evaluate via the same outer CV splitter you already chose. **Anti-overfit guard**: pick the winner by mean CV score on **the same outer folds used for the rest of the grid**, with the held-out fold ALWAYS being a fresh fold not used to make any prior decision (e.g. don't pick threshold + class_weight on the same OOF predictions). If the metric is F1 (threshold-sensitive), use `predict_proba` and a **fixed 0.5 threshold** for evaluation — do NOT also tune the threshold here (that's the past `skill_tune_decision_threshold_for_f1` overfit pattern).

Persist `result.json.audit.class_weight_grid = [...]` and `chosen_class_weight = ...`.

```python
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

variants = [
    {"sample_weight": None, "tag": "no_weight"},
    {"sample_weight": "balanced", "tag": "balanced"},
]
cv_scores = {}
for v in variants:
    fold_scores = []
    for tr_idx, va_idx in cv.split(X, y):
        sw = compute_sample_weight("balanced", y[tr_idx]) if v["sample_weight"] == "balanced" else None
        m = HistGradientBoostingClassifier(**best_params, random_state=0).fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        proba = m.predict_proba(X[va_idx])[:, 1]
        fold_scores.append(scoring(y[va_idx], proba))   # AUC if AUC, F1 with thr=0.5 if F1
    cv_scores[v["tag"]] = float(np.mean(fold_scores))
```

## Why
H_0 baseline failures on the 20-task pool:
- `playground-series-s3e3` (Attrition, 12% positive, AUC 0.819): `skill_tune_when_n_train_supports_it` mentions `class_weight` but the solver did NOT include it in the grid; the chosen `GradientBoostingClassifier` was trained without weighting.
- `playground-series-s3e23` (Defects AUC 0.788, ~23% minority — borderline) and other `roc_auc`-binary tasks may similarly leave easy gains on the table.

Class weighting is metric-aware — for AUC it usually has a tiny effect on rank but can shift logloss-trained gradient boosters' early-iteration calibration; for F1/accuracy under heavy imbalance it can matter several points. Including it as a **second-axis grid** with the existing tuning skill is cheap (≤2x configs) and the choice is made by held-out CV, not by maximizing on the same OOF used to choose anything else.

## Anti-overfit note (vs the past F1-threshold failure)
The previous overfit pattern was tuning a **decision threshold** on the SAME OOF predictions used to compute headline F1, which moved CV-F1 up but hurt native test F1. This skill does NOT tune a threshold; it tunes a model-fitting setting (`class_weight` / `sample_weight`) under a fixed threshold (0.5 for F1, threshold-free for AUC). The selection happens on held-out folds, not by re-fitting probabilities to a target metric on the OOF.
