"""Solution for playground-series-s3e7 (Reservation Cancellation Prediction).

Binary classification, metric=ROC AUC. n_train=33680.

Skills applied:
- skill_tune_when_n_train_supports_it (n>=500, no constraint tokens) — small 3-config grid
  over HistGradientBoostingClassifier.
- skill_class_weight_for_imbalanced_binary precondition FAILS (minority rate ~0.39 > 0.20),
  so we do NOT add class_weight axis.
- skill_stringify_low_cardinality_int_codes precondition FAILS (tree model, not linear/distance).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e7/data")
TARGET = "booking_status"
ID = "id"
SEED = 0
N_JOBS = 2

t0 = time.time()
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

feature_cols = [c for c in train.columns if c not in (ID, TARGET)]
X = train[feature_cols].values.astype(np.float32)
y = train[TARGET].values.astype(np.int8)
X_test = test[feature_cols].values.astype(np.float32)

# --- Skill: tune_when_n_train_supports_it — small explicit grid ---
grid = [
    dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0),
    dict(max_iter=500, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=1.0),
    dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=1.0),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = []
for params in grid:
    fold_scores = []
    for tr_idx, va_idx in cv.split(X, y):
        m = HistGradientBoostingClassifier(**params, random_state=SEED, early_stopping=False)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[va_idx])[:, 1]
        fold_scores.append(roc_auc_score(y[va_idx], p))
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    results.append((mean, std, params, fold_scores))
    print(f"  cfg={params}: AUC={mean:.5f} ± {std:.5f}")

results.sort(key=lambda r: r[0], reverse=True)
best_mean, best_std, best_params, best_folds = results[0]
print(f"BEST: {best_params}  CV-AUC={best_mean:.5f} ± {best_std:.5f}")

# --- Refit on full train ---
final_model = HistGradientBoostingClassifier(**best_params, random_state=SEED, early_stopping=False)
final_model.fit(X, y)
test_pred = final_model.predict_proba(X_test)[:, 1]

# --- Submission ---
sub = pd.DataFrame({ID: test[ID].values, TARGET: test_pred})
sub.to_csv(HERE / "submission.csv", index=False)

# --- Result JSON ---
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "roc_auc",
    "headline_value": best_mean,
    "all_metrics": {
        "cv_roc_auc_mean": best_mean,
        "cv_roc_auc_std": best_std,
        "fold_scores": best_folds,
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
    ],
    "interpretation_notes": (
        "HistGradientBoosting on raw features; 3-config CV grid; "
        "n=33680 supports tuning. Class-weight skill skipped (minority=0.39 > 0.20). "
        "Stringify-int-codes skill skipped (tree model)."
    ),
    "audit": {
        "hyperparameter_grid": [r[2] for r in results],
        "chosen_config": best_params,
        "grid_scores": [{"params": r[2], "mean": r[0], "std": r[1]} for r in results],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "elapsed_sec": time.time() - t0,
    },
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2, default=float)

print(f"Wrote submission.csv ({len(sub)} rows) and result.json. Elapsed={time.time()-t0:.1f}s")
