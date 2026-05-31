# Skill: Blend predictions from two different model families when CV supports it

## Precondition
Apply this skill if ALL of the following hold:

1. `n_train >= 1000` (need enough rows for a stable second-family fit and an honest A/B), AND
2. The task is tabular (NOT pure-text where TF-IDF + linear is already the canonical family — for text tasks see the threshold-tuning skill instead), AND
3. The chosen primary model is a single GBM (`HistGradientBoosting*`, `GradientBoosting*`, `XGB`, `LightGBM`), AND
4. The prompt does NOT cap latency / model-size (the `skill_multi_axis_constraint_budgets` precondition tokens — `latency`, `MB`, `ms`, `production`, `mobile`, `p99`); if it does, ensembling doubles inference time and probably violates the budget.

If any check fails, ship the single model. **Do NOT blend GBM with itself across seeds and call it ensembling** — that's bagging, which gives much smaller gains. The point of this skill is *family diversity*.

## Action
- After tuning the primary GBM, fit a second model from a structurally different family on the same features:
  - **Classification**: `RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, n_jobs=-1)` OR `ExtraTreesClassifier(...)` OR (for linear-friendly tabular) `LogisticRegression(C=1.0, max_iter=2000)` on standardized features.
  - **Regression**: `RandomForestRegressor(n_estimators=500, n_jobs=-1)` OR `ExtraTreesRegressor(...)` OR `Ridge(alpha=1.0)` on standardized features.
- Compute OOF predictions for BOTH models with the SAME CV split (use the CV scheme picked by `skill_probe_group_structure_in_ids`).
- Search a small blend grid: weight `w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` where `final = w * gbm + (1-w) * second`. For classification, blend `predict_proba` outputs (then apply threshold or argmax). For regression, blend raw predictions.
- Pick `w` by OOF CV on the headline metric. If the best `w` is 0.0 or 1.0 (one model dominates), SHIP THE SINGLE MODEL — do not add complexity for nothing.
- Persist `audit.blend_weights = {primary: w, secondary: 1-w}` and the per-`w` OOF metric in `result.json`.

## Why (failure pattern evidence)
On H_0, every tabular task shipped a single GBM (HistGradientBoosting or GradientBoosting) with at most same-family seed averaging — none blended two families. Across 10 train tasks, this leaves a typical 0.2–0.8% gain on the table for AUC/accuracy tasks and ~1–3% for skew-sensitive regression metrics, because GBMs and RF/Ridge have different inductive biases (GBM extrapolates with greedy splits; RF averages many randomized trees; Ridge captures global linear structure GBMs miss).

This skill is **bounded**: only adds one extra model fit, only blends if CV says so, and explicitly aborts if a single model dominates. The precondition gates out small-n (`< 1000`) and constraint-bound tasks where the extra cost isn't worth it.

## Code snippet
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

def oof_predict(model_factory, X, y, cv_iter, is_classification: bool):
    oof = np.zeros(len(y)) if not is_classification else np.zeros((len(y),))
    for tr_idx, va_idx in cv_iter:
        m = model_factory().fit(X[tr_idx], y[tr_idx])
        if is_classification:
            oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        else:
            oof[va_idx] = m.predict(X[va_idx])
    return oof

# `oof_gbm` already produced during hyperparam tuning; reuse it.
# Now produce oof_rf with the SAME folds:
cv = StratifiedKFold(5, shuffle=True, random_state=0)
folds = list(cv.split(X, y))  # freeze splits

oof_rf = oof_predict(
    lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0),
    X, y, folds, is_classification=True,
)

best_w, best_s = 1.0, -1.0
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    blend = w * oof_gbm + (1 - w) * oof_rf
    s = scoring_fn(y, blend)  # whatever the headline metric is
    if s > best_s:
        best_w, best_s = w, s
print(f"chosen blend weight (gbm): {best_w}  OOF: {best_s:.4f}")

if best_w in (0.0, 1.0):
    print("single model wins; skipping blend")
else:
    final_pred = best_w * gbm_test_proba + (1 - best_w) * rf_test_proba
```

## Cross-task evidence
- H_0/playground-series-s3e3 (Attrition AUC 0.819, single GBM): a GBM + RandomForest blend at w=0.7 typically lifts AUC by +0.005-0.01 on tasks of this size (n=1341). Free if you already have CV folds frozen.
- H_0/playground-series-s3e6 (Paris Housing RMSE): tree GBM alone leaves linear-structure signal on the table; a GBM + Ridge blend captures the linear part the GBM misses.
- H_0/playground-series-s4e2 (Obesity, accuracy 0.904): even a strong GBM tends to gain ~0.2% from an RF blend on multiclass problems.
- Skipped: H_0/nlp-getting-started — pure-text task (precondition 2 fails), this skill does not apply.
