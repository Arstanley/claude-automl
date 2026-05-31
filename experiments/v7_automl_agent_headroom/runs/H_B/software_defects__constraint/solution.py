"""Software Defects binary classification.

Skills applied:
  - skill_multi_axis_constraint_budgets: 30min train + 5ms/inst inference budgets.
    HistGradientBoostingClassifier is ~us/inst at predict time; fits in seconds.
  - skill_verify_split_claim_before_holdout: prompt asks for 3-way split, but
    real test set is provided separately. Use stratified 80/20 of train for a
    held-out valid set used only for F1-threshold tuning.

Skipped:
  - skill_tune_when_n_train_supports_it: precondition explicitly excludes
    prompts that mention "latency / ms / production". This task does, so we
    keep a tiny grid (3 configs) only.
  - skill_missing_indicator: zero NaNs in train; precondition fails.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/software_defects/data")
N_JOBS = 2
SEED = 42

t0 = time.time()

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

TARGET = "defects"
ID = "id"
y = train[TARGET].astype(int).values
X = train.drop(columns=[TARGET, ID])
X_test = test.drop(columns=[ID])
feat_cols = list(X.columns)

# Tiny constraint-aware grid: 3 configs.
grid = [
    {"max_iter": 400, "learning_rate": 0.05, "max_depth": None, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_iter": 600, "learning_rate": 0.05, "max_depth": None, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_iter": 400, "learning_rate": 0.08, "max_depth": None, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
]

# Hold out 20% stratified for threshold tuning + reporting.
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
sw_tr = compute_sample_weight("balanced", y_tr)

def fit_eval(params):
    m = HistGradientBoostingClassifier(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        **params,
    )
    m.fit(X_tr, y_tr, sample_weight=sw_tr)
    p = m.predict_proba(X_val)[:, 1]
    # F1 threshold sweep
    thr = np.linspace(0.05, 0.95, 91)
    best = max(((f1_score(y_val, (p >= t).astype(int)), t) for t in thr), key=lambda x: x[0])
    return m, p, best  # (model, val_proba, (f1, thr))

best = None
all_results = []
for params in grid:
    m, p, (f1v, thr) = fit_eval(params)
    all_results.append({"params": params, "val_f1": float(f1v), "thr": float(thr)})
    if best is None or f1v > best[0]:
        best = (f1v, thr, m, params, p)

best_f1, best_thr, best_model, best_params, best_val_p = best

# Refit on full train with chosen params for final submission.
sw_full = compute_sample_weight("balanced", y)
final = HistGradientBoostingClassifier(
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    **best_params,
)
final.fit(X, y, sample_weight=sw_full)

# Measure inference latency on a 1000-row sample.
sample = X_test.iloc[: min(1000, len(X_test))]
t_lat = time.time()
_ = final.predict_proba(sample)
lat_per_inst_ms = 1000.0 * (time.time() - t_lat) / len(sample)

p_test = final.predict_proba(X_test)[:, 1]
yhat = (p_test >= best_thr).astype(int)

sub = pd.DataFrame({ID: test[ID].values, TARGET: yhat})
sub.to_csv(ROOT / "submission.csv", index=False)

# Compute val metrics at chosen threshold for reporting.
val_pred = (best_val_p >= best_thr).astype(int)
metrics = {
    "val_f1": float(f1_score(y_val, val_pred)),
    "val_precision": float(precision_score(y_val, val_pred)),
    "val_recall": float(recall_score(y_val, val_pred)),
    "val_auc": float(roc_auc_score(y_val, best_val_p)),
    "threshold": float(best_thr),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "positive_rate_train": float(y.mean()),
    "positive_rate_pred": float(yhat.mean()),
    "inference_ms_per_instance": float(lat_per_inst_ms),
    "wall_seconds": float(time.time() - t0),
    "grid_results": all_results,
    "chosen_params": best_params,
}

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "f1",
    "headline_value": float(best_f1),
    "all_metrics": metrics,
    "skills_applied": [
        "skill_multi_axis_constraint_budgets.md",
        "skill_verify_split_claim_before_holdout.md",
    ],
    "interpretation_notes": (
        "HistGradientBoosting with class-balanced sample weights; 3-config grid "
        "(latency-token in prompt suppresses heavier tuning). Threshold tuned on "
        "20% stratified hold-out to maximize F1 on the positive (defective) class."
    ),
}

with open(ROOT / "result.json", "w") as f:
    json.dump(result, f, indent=2, default=float)

print("DONE  val_f1={:.4f}  thr={:.3f}  lat={:.4f} ms/inst  wall={:.1f}s".format(
    best_f1, best_thr, lat_per_inst_ms, time.time() - t0
))
