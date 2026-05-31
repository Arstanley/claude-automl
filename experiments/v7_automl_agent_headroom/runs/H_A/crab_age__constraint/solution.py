"""
Crab Age regression — RMSLE.

Skills applied:
  - skill_tune_when_n_train_supports_it: n_train=59,240 (>>500); constraints are
    generous (30 min train, 5 s predict — no "ms"/"MB"/"latency" tokens) so a
    small explicit GBM grid is appropriate.

Approach:
  - HistGradientBoostingRegressor on log1p(Age) target (RMSLE-aware).
  - Sex one-hot encoded; numeric features passed through.
  - 5-fold CV over a 3-config grid; report mean RMSLE-on-log space.
  - Refit best config on full train; predict on test; clip to >=1.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/crab_age/data")
N_JOBS = 2

t0 = time.time()
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

TARGET = "Age"
ID = "id"
NUMERIC = ["Length", "Diameter", "Height", "Weight", "Shucked Weight",
           "Viscera Weight", "Shell Weight"]

# One-hot encode Sex (only 3 categories: M/F/I)
def featurize(df):
    X = df[NUMERIC].copy()
    sex = df["Sex"].astype(str)
    for v in ("M", "F", "I"):
        X[f"Sex_{v}"] = (sex == v).astype(np.int8)
    return X.values.astype(np.float32)

X_tr = featurize(train)
X_te = featurize(test)
y_log = np.log1p(train[TARGET].values.astype(np.float64))

print(f"Loaded train={X_tr.shape}, test={X_te.shape} in {time.time()-t0:.1f}s")

# Small explicit grid (3 configs) per the tune-when-n-supports-it skill.
grid = [
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
         l2_regularization=0.0, min_samples_leaf=20),
    dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=63,
         l2_regularization=0.1, min_samples_leaf=20),
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=15,
         l2_regularization=0.0, min_samples_leaf=40),
]

def rmsle_from_logspace(y_log_true, y_log_pred):
    # We trained on log1p(y); pred in same space. RMSLE = sqrt(mean((log(1+yhat) - log(1+y))^2))
    # = sqrt(mean((y_log_pred - y_log_true)^2)) since both already log1p.
    return float(np.sqrt(np.mean((y_log_pred - y_log_true) ** 2)))

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_results = []
for params in grid:
    fold_scores = []
    for tr_idx, va_idx in cv.split(X_tr):
        m = HistGradientBoostingRegressor(random_state=0, **params)
        m.fit(X_tr[tr_idx], y_log[tr_idx])
        pred = m.predict(X_tr[va_idx])
        pred = np.clip(pred, np.log1p(1.0), None)  # Age >= 1
        fold_scores.append(rmsle_from_logspace(y_log[va_idx], pred))
    mean, std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f"  {params} -> RMSLE {mean:.5f} +/- {std:.5f}")
    cv_results.append({"params": params, "mean": mean, "std": std})

cv_results.sort(key=lambda r: r["mean"])
best = cv_results[0]
print("Best:", best)

# Refit on full data
t_fit = time.time()
final = HistGradientBoostingRegressor(random_state=0, **best["params"])
final.fit(X_tr, y_log)
print(f"Refit on full train in {time.time()-t_fit:.1f}s")

t_pred = time.time()
pred_log = final.predict(X_te)
pred_log = np.clip(pred_log, np.log1p(1.0), None)
pred = np.expm1(pred_log)
pred = np.clip(pred, 1.0, None)
pred_secs = time.time() - t_pred
print(f"Predicted test ({len(pred)} rows) in {pred_secs:.3f}s")

# Write submission
sub = pd.DataFrame({ID: test[ID].values, TARGET: pred})
sub.to_csv(HERE / "submission.csv", index=False)

# Result json
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "RMSLE",
    "headline_value": best["mean"],
    "all_metrics": {
        "cv_rmsle_mean": best["mean"],
        "cv_rmsle_std": best["std"],
        "predict_seconds": pred_secs,
    },
    "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
    "interpretation_notes": (
        "HistGradientBoostingRegressor on log1p(Age) (RMSLE-aware); "
        "3-config small explicit grid with 5-fold CV per skill_tune_when_n_train_supports_it "
        "(n_train=59240, no sub-100ms/MB/latency constraints). Best config refit on full train."
    ),
    "audit": {
        "hyperparameter_grid": [r["params"] for r in cv_results],
        "cv_results": cv_results,
        "chosen_config": best["params"],
    },
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Total wall time: {time.time()-t0:.1f}s")
print(f"submission.csv rows: {len(sub)}; first row: {sub.iloc[0].to_dict()}")
