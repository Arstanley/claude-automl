"""
Solution for playground-series-s3e14 (Wild Blueberry Yield Prediction).

Task: Regression, metric = MAE.
Skills applied:
  - skill_tune_when_n_train_supports_it (n_train=12231, no size/latency tokens)
  - skill_seed_bag_final_regressor (regression, n>=500, no latency tokens)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e14/data"
OUT_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/runs/iter1_e1/playground-series-s3e14"

TARGET = "yield"
ID = "id"
N_JOBS = 2

t0 = time.time()

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

feature_cols = [c for c in train.columns if c not in (TARGET, ID)]
X = train[feature_cols].values
y = train[TARGET].values
X_test = test[feature_cols].values

# CV setup — KFold(5) sufficient (no group structure: train and test ids are disjoint
# but there's no compound id; tabular regression)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Small grid (skill_tune_when_n_train_supports_it: 3-5 configs)
grid = [
    {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"learning_rate": 0.03, "max_iter": 800, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0},
]

results = []
for cfg in grid:
    model = HistGradientBoostingRegressor(
        loss="absolute_error",  # match MAE objective
        early_stopping=False,
        random_state=0,
        **cfg,
    )
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=N_JOBS
    )
    mae = -scores.mean()
    results.append((mae, cfg))
    print(f"cfg={cfg} CV-MAE={mae:.4f} (elapsed {time.time()-t0:.1f}s)")

results.sort(key=lambda r: r[0])
best_mae, best_cfg = results[0]
print(f"BEST cfg={best_cfg} CV-MAE={best_mae:.4f}")

# Seed bag (skill_seed_bag_final_regressor): refit on full train with 3 seeds.
seeds = [0, 1, 2]
test_preds = []
for s in seeds:
    m = HistGradientBoostingRegressor(
        loss="absolute_error",
        early_stopping=False,
        random_state=s,
        **best_cfg,
    )
    m.fit(X, y)
    test_preds.append(m.predict(X_test))
final_pred = np.mean(test_preds, axis=0)

# Submission keyed to test.csv ids (per task spec: "for each id in the test set")
sub = pd.DataFrame({ID: test[ID].values, TARGET: final_pred})
sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "MAE",
    "headline_value": float(best_mae),
    "all_metrics": {
        "cv_mae_best": float(best_mae),
        "cv_mae_all": [{"cfg": cfg, "mae": float(mae)} for mae, cfg in results],
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "HistGradientBoostingRegressor with loss='absolute_error' (matches MAE metric); "
        "small 4-config grid tuned via KFold(5) CV; final predictions are mean of 3 seed-bagged refits. "
        "All features numeric and low-cardinality with no NaNs — no preprocessing needed."
    ),
    "audit": {
        "seed_bag": seeds,
        "best_cfg": best_cfg,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "elapsed_s": time.time() - t0,
    },
}
with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"Done in {time.time()-t0:.1f}s. submission.csv rows={len(sub)}")
