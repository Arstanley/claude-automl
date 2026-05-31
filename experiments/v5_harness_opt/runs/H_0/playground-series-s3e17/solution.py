"""
Kaggle Playground Series S3E17 — Machine Failure (binary classification, ROC AUC).

Strategy:
- HistGradientBoostingClassifier (fast on 109K rows, n_jobs irrelevant: native parallel).
- Features: numeric sensor readings + Type one-hot + 5 binary failure-mode flags
  (TWF/HDF/PWF/OSF/RNF). Product ID is high-cardinality with heavy train/test overlap
  but appears random — drop it to keep generalization clean.
- Small 2-config grid over learning_rate / max_iter selected by 3-fold stratified CV
  AUC (skill_tune_when_n_train_supports_it: n_train=109K, no size/latency constraint).
- Final model refit on all train data with best config; output probabilities to submission.csv.
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

HERE = Path(__file__).parent.resolve()
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e17/data")
TARGET = "Machine failure"
ID_COL = "id"
N_JOBS = 2
SEED = 42

t0 = time.time()

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

feat_num = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
feat_bin = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Type -> one-hot (low-cardinality categorical)
full = pd.concat([train.assign(_src="tr"), test.assign(_src="te")], ignore_index=True, sort=False)
full = pd.get_dummies(full, columns=["Type"], prefix="Type", dtype=np.int8)
type_cols = [c for c in full.columns if c.startswith("Type_")]

feat_cols = feat_num + feat_bin + type_cols

X_tr = full.loc[full["_src"] == "tr", feat_cols].reset_index(drop=True).astype(np.float32)
X_te = full.loc[full["_src"] == "te", feat_cols].reset_index(drop=True).astype(np.float32)
y_tr = train[TARGET].astype(np.int8).values
test_ids = test[ID_COL].values

print(f"X_tr={X_tr.shape}  X_te={X_te.shape}  pos rate={y_tr.mean():.4f}", flush=True)

# Small grid (<=2 configs) per harness budget
grid = [
    {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 1.0},
]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

cv_results = []
for i, params in enumerate(grid):
    fold_aucs = []
    for fold, (tr_i, va_i) in enumerate(skf.split(X_tr, y_tr)):
        model = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=SEED,
        )
        model.fit(X_tr.iloc[tr_i], y_tr[tr_i])
        p = model.predict_proba(X_tr.iloc[va_i])[:, 1]
        auc = roc_auc_score(y_tr[va_i], p)
        fold_aucs.append(auc)
        print(f"  cfg{i} fold{fold} AUC={auc:.5f}", flush=True)
    mean_auc = float(np.mean(fold_aucs))
    cv_results.append({"params": params, "fold_aucs": fold_aucs, "mean_auc": mean_auc})
    print(f"cfg{i} mean AUC={mean_auc:.5f}  elapsed={time.time()-t0:.1f}s", flush=True)

best = max(cv_results, key=lambda r: r["mean_auc"])
print("BEST:", best["params"], "AUC=", best["mean_auc"], flush=True)

# Refit on full train
final = HistGradientBoostingClassifier(
    **best["params"],
    early_stopping=False,
    random_state=SEED,
)
final.fit(X_tr, y_tr)
proba = final.predict_proba(X_te)[:, 1]

sub = pd.DataFrame({ID_COL: test_ids, TARGET: proba})
sub.to_csv(HERE / "submission.csv", index=False)
print(f"wrote submission.csv shape={sub.shape}", flush=True)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "roc_auc",
    "headline_value": best["mean_auc"],
    "all_metrics": {
        "cv_mean_auc": best["mean_auc"],
        "cv_fold_aucs": best["fold_aucs"],
        "grid": [{"params": r["params"], "mean_auc": r["mean_auc"]} for r in cv_results],
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_stringify_low_cardinality_int_codes.md",
    ],
    "interpretation_notes": (
        "HistGradientBoosting on 109K rows with Type one-hot and the 5 failure-mode flags + 5 sensors. "
        "Dropped high-card Product ID (random split; no group signal warranted). "
        "Tuned a 2-config grid (lr/leaves/min_samples_leaf) via 3-fold stratified CV; refit on full train."
    ),
    "runtime_seconds": round(time.time() - t0, 2),
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"wrote result.json  total={time.time()-t0:.1f}s", flush=True)
