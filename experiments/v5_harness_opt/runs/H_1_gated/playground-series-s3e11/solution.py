"""
Solution for playground-series-s3e11 (Media Campaign Cost Prediction).
Tabular regression, RMSLE metric.

Skills applied:
  - skill_tune_when_n_train_supports_it (n_train=288k, no size/latency constraints)
  - skill_seed_bag_final_regressor (regression, n>=500, no latency tokens)
"""
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

os.environ.setdefault("OMP_NUM_THREADS", "2")

T0 = time.time()
DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e11/data"
WORK_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1_gated/playground-series-s3e11"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET = "cost"
ID = "id"
features = [c for c in train.columns if c not in (TARGET, ID)]

X = train[features].values
y = train[TARGET].values
X_test = test[features].values
y_log = np.log1p(y)

# Small explicit grid (skill_tune_when_n_train_supports_it).
# Use 3-fold CV on a subsample for speed within the 4-min budget.
rng = np.random.default_rng(0)
n_sub = min(len(X), 80_000)
sub_idx = rng.choice(len(X), n_sub, replace=False)
Xs, ys = X[sub_idx], y_log[sub_idx]

grid = [
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0, min_samples_leaf=20),
    dict(max_iter=500, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=1.0, min_samples_leaf=20),
    dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=63, l2_regularization=1.0, min_samples_leaf=40),
]

cv = KFold(n_splits=3, shuffle=True, random_state=0)
cv_results = []
for i, params in enumerate(grid):
    oof = np.zeros(len(Xs))
    for tr, va in cv.split(Xs):
        m = HistGradientBoostingRegressor(**params, random_state=0, early_stopping=False)
        m.fit(Xs[tr], ys[tr])
        oof[va] = m.predict(Xs[va])
    # RMSLE on original scale
    pred_y = np.clip(np.expm1(oof), 0, None)
    true_y = np.expm1(ys)
    rmsle = float(np.sqrt(mean_squared_log_error(true_y, pred_y)))
    cv_results.append((rmsle, params))
    print(f"config {i}: RMSLE={rmsle:.5f} params={params}")

cv_results.sort(key=lambda r: r[0])
best_rmsle_sub, best_params = cv_results[0]
print(f"Best subsample CV RMSLE: {best_rmsle_sub:.5f}")
print(f"Best params: {best_params}")
print(f"Elapsed after CV: {time.time()-T0:.1f}s")

# Seed bag final regressor (skill_seed_bag_final_regressor).
seeds = [0, 1, 2]
test_preds_log = []
for s in seeds:
    m = HistGradientBoostingRegressor(**best_params, random_state=s, early_stopping=False)
    m.fit(X, y_log)
    test_preds_log.append(m.predict(X_test))
    print(f"seed {s} done at {time.time()-T0:.1f}s")

final_log = np.mean(test_preds_log, axis=0)
final_pred = np.clip(np.expm1(final_log), 0.0, None)

sub = pd.DataFrame({ID: test[ID].values, "cost": final_pred})
sub_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"Wrote {sub_path}")

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "RMSLE",
    "headline_value": best_rmsle_sub,
    "all_metrics": {
        "RMSLE_subsample_cv": best_rmsle_sub,
        "grid_rmsle": [r[0] for r in cv_results],
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "Tabular regression with RMSLE: trained HistGradientBoostingRegressor on log1p(cost) "
        "(RMSLE = RMSE on log1p). Selected one of 3 configs via 3-fold CV on an 80k subsample for "
        "speed under the 4-min budget, then refit on full train with 3 random seeds (seed bag) and "
        "averaged predictions for variance reduction."
    ),
    "audit": {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "hyperparameter_grid": [r[1] for r in cv_results],
        "chosen_config": best_params,
        "seed_bag": seeds,
        "elapsed_sec": time.time() - T0,
    },
}
res_path = os.path.join(WORK_DIR, "result.json")
with open(res_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"Wrote {res_path}")
print(f"Total elapsed: {time.time()-T0:.1f}s")
