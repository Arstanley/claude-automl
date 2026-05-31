# Skill: Tune at least the headline hyperparameters; don't ship sklearn defaults

## When this applies
Final-model selection on any non-trivial dataset. Sklearn / sklearn-style libraries ship with conservative defaults that are rarely optimal for tabular ML, and "default model + clean pipeline" loses to "modestly tuned model + clean pipeline" by 1-3pp routinely.

## What to do
- Decide model family from the problem shape:
  - Small tabular (<100k rows), mixed numeric/categorical → gradient boosting (`GradientBoostingClassifier`, `HistGradientBoosting*`, LightGBM, XGBoost) or RandomForest. These dominate.
  - Want calibrated probabilities → GBM with log-loss, or RF + calibration. Avoid hard-classifier-only models when the metric is AUC/log-loss.
  - High-dimensional sparse → linear / ElasticNet.
- For GBM/RF, at minimum tune (in rough order of impact): `n_estimators` (often 300-1000 vs default 100), `learning_rate` (0.03-0.1 vs default 0.1), `max_depth` (3-8 vs default unrestricted/3), `min_samples_leaf` / `min_child_samples`, `subsample` / `colsample_bytree` (0.7-0.9), regularization.
- Use a small grid or coarse random search (10-30 configs) with the **same CV protocol** you decided on; don't tune on a single holdout.
- Use **early stopping** when the library supports it (XGBoost, LightGBM, HistGradientBoosting `early_stopping_rounds`) — this lets `n_estimators` be set generously without overfit risk.
- For classification with imbalance, set `class_weight="balanced"` (RF, linear) or `scale_pos_weight` (XGB/LGB), OR explicitly choose not to and report why.
- Don't over-tune: with small N, hyperparameter selection variance can exceed the gain. Pick a sensible config, lock it in, and document.
- Always re-fit the tuned model on the full train set before making the test predictions (not on the CV folds).

## Why
Defaults are a calibrated guess for "any problem" — your problem has structure (size, feature count, imbalance, noise) that defaults don't know. Saw on s3e3: C_warned lost by 2pp largely because it used `n_estimators=100, max_depth=3, learning_rate=0.1` defaults while A and B used `n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9`.

## Code snippet
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Small explicit grid (not a giant search)
grid = [
    dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9),
    dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.9),
    dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8),
]
best_score, best_params = -np.inf, None
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for params in grid:
    fold_scores = []
    for tr, va in cv.split(X, y):
        m = GradientBoostingClassifier(**params, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr])
        fold_scores.append(score_fn(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    mean = np.mean(fold_scores)
    print(f"{params}: {mean:.4f} ± {np.std(fold_scores):.4f}")
    if mean > best_score:
        best_score, best_params = mean, params

# Refit on full train with best params
final = GradientBoostingClassifier(**best_params, random_state=0).fit(X, y)
preds = final.predict_proba(X_test)[:, 1]
```

## Cross-task evidence
- Saw on: playground-series-s3e3 (C_warned used sklearn defaults — AUC 0.8153 vs A/B's tuned 0.832/0.840; the 2pp gap was the difference between winning and losing), playground-series-s3e22 (B used class_weight=balanced_subsample which helped on the imbalanced 3-class problem), bike-sharing-demand (B and C tuned GBM trees; A used a simpler config), titanic and spaceship-titanic (less hyperparameter-sensitive but still benefit from non-default n_estimators).
