# Skill: Train on log1p(target) when the target is right-skewed and non-negative

## Precondition
Apply this skill if ALL of the following hold:

1. The task is regression (the headline metric is one of `rmse`, `mae`, `rmsle`, `mape`, `r2`, or a regression metric), AND
2. `(train[target] >= 0).all()` — all training targets are non-negative, AND
3. EITHER the headline metric is `rmsle` / `msle` / contains the substring `log` (mechanical match),
   OR `train[target].skew() >= 1.0` (right-skewed; sklearn / pandas `.skew()`),
   OR `train[target].max() / max(1.0, train[target].median()) >= 20` (long-tailed: max is ≥20× median).

If any check fails (negative targets, symmetric distribution, log already in the metric definition with a custom transform you've handled), skip this skill.

## Action
- Fit the model on `y_log = np.log1p(y)` instead of raw `y`.
- At predict time: `y_pred = np.clip(np.expm1(model.predict(X_test)), 0, None)`.
- When the metric is `rmsle`, fitting on `log1p(y)` with plain MSE/RMSE loss is **equivalent** to optimizing RMSLE directly — this is the textbook trick, and it's the whole point of the transform for that metric.
- When the metric is `rmse`/`mae` on a heavy-tailed target, log-fitting reduces leverage of the largest targets and usually wins by 1-15% RMSE on right-skewed data (housing prices, counts, durations, concentrations).
- Compare the OOF score of the log-fit model vs a raw-y baseline in a quick A/B; if raw-y wins by more than 1 std on CV, keep raw-y. Persist both in `result.json.audit.log_target_ab` for auditability.
- Persist `audit.target_transform = "log1p"` (or `"identity"`) and the CV-mean of each.

## Why (failure pattern evidence)
On H_0/playground-series-s3e6 (Paris Housing, RMSE metric, target `price` is right-skewed with max ≈ 5M and median ≈ 700k), the solver fit HistGradientBoostingRegressor on raw `price` and reported CV RMSE 151,588. Fitting on `log1p(price)` (then expm1 back) typically beats this by 5-10% RMSE on Paris-Housing-style data because the model no longer over-weights a handful of mega-priced outliers.

On H_0/playground-series-s3e9 (Concrete Strength, RMSE), target is non-negative and right-skewed (concrete strength has a long right tail of high-strength mixes). The solver fit on raw strength; a log-fit comparison was not even attempted. The skill's A/B clause forces the comparison so the better choice wins by CV, not by guess.

On H_0/bike-sharing-demand the solver DID apply `log1p(count)` (correctly, since the metric is RMSLE) — proof that the transform pattern works; this skill generalizes the same trick to any right-skewed regression target.

## Code snippet
```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def should_log_transform(y: np.ndarray, metric_name: str) -> bool:
    if (y < 0).any():
        return False
    if "log" in metric_name.lower() or "msle" in metric_name.lower():
        return True
    try:
        sk = float(((y - y.mean())**3).mean() / (y.std() ** 3 + 1e-9))
    except Exception:
        sk = 0.0
    return sk >= 1.0 or (y.max() / max(1.0, np.median(y)) >= 20)

USE_LOG = should_log_transform(y, headline_metric)
print(f"target transform: {'log1p' if USE_LOG else 'identity'}")

def fit_predict(X_tr, y_tr, X_te):
    if USE_LOG:
        m = Model(**best_params).fit(X_tr, np.log1p(y_tr))
        return np.clip(np.expm1(m.predict(X_te)), 0, None)
    else:
        m = Model(**best_params).fit(X_tr, y_tr)
        return m.predict(X_te)

# A/B check on CV (sanity, only ~30s extra)
cv = KFold(5, shuffle=True, random_state=0)
log_scores, raw_scores = [], []
for tr_idx, va_idx in cv.split(X):
    log_pred = fit_predict_with_flag(True, X[tr_idx], y[tr_idx], X[va_idx])
    raw_pred = fit_predict_with_flag(False, X[tr_idx], y[tr_idx], X[va_idx])
    log_scores.append(np.sqrt(mean_squared_error(y[va_idx], log_pred)))
    raw_scores.append(np.sqrt(mean_squared_error(y[va_idx], raw_pred)))
USE_LOG = np.mean(log_scores) < np.mean(raw_scores)
```

## Cross-task evidence
- H_0/bike-sharing-demand: solver correctly log1p-modeled count; metric is RMSLE so this is mandatory. Got 0.260 RMSLE.
- H_0/playground-series-s3e6: solver did NOT log-transform price (right-skewed); skill would force the A/B and likely flip to log.
- H_0/playground-series-s3e9: solver did NOT log-transform strength; skill would force the A/B.
