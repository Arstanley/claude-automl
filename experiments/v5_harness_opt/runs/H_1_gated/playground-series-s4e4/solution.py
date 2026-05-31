"""
Solution for playground-series-s4e4 (Abalone Rings regression, RMSLE).

Skills applied:
- skill_tune_when_n_train_supports_it.md (n_train=72492 >> 500, no latency/size
  tokens, GBM family) -> small 3-config grid evaluated by CV RMSLE.
- skill_seed_bag_final_regressor.md (regression with nunique(target)=28 > 20,
  n>=500, model takes random_state, no constraint tokens) -> 3-seed bag on
  full-train refits.

Modeling notes:
- RMSLE => predict in log1p space; back-transform with expm1.
- One-hot encode `Sex` (M/F/I). Other features numeric.
- Use HistGradientBoostingRegressor (fast, handles ~72k rows comfortably).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

os.environ.setdefault("OMP_NUM_THREADS", "2")

DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e4/data/"
WORK = "/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1_gated/playground-series-s4e4/"

t0 = time.time()
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET = "Rings"
ID = "id"
features = [c for c in train.columns if c not in (TARGET, ID)]
cat_cols = ["Sex"]
num_cols = [c for c in features if c not in cat_cols]

X = train[features].copy()
y = train[TARGET].astype(float).values
y_log = np.log1p(y)
X_test = test[features].copy()


def make_pipeline(**hgb_params):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(loss="squared_error", **hgb_params)
    return Pipeline([("pre", pre), ("model", model)])


def rmsle_from_log(y_true_log, y_pred_log):
    # y_true, y_pred guaranteed nonneg after expm1+clip; equivalent to RMSE in log1p space
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


# ---- skill_tune_when_n_train_supports_it: 3-config grid, CV by RMSLE ----
grid = [
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
    dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.0),
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
]

cv = KFold(n_splits=3, shuffle=True, random_state=0)
cv_results = []
for params in grid:
    fold_scores = []
    for tr_idx, va_idx in cv.split(X):
        pipe = make_pipeline(random_state=0, **params)
        pipe.fit(X.iloc[tr_idx], y_log[tr_idx])
        pred_log = pipe.predict(X.iloc[va_idx])
        pred_log = np.maximum(pred_log, 0.0)  # ensure nonneg in log space
        fold_scores.append(rmsle_from_log(y_log[va_idx], pred_log))
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    cv_results.append({"params": params, "mean_rmsle": mean, "std_rmsle": std})
    print(f"  {params} -> RMSLE {mean:.5f} ± {std:.5f}")

cv_results.sort(key=lambda r: r["mean_rmsle"])
best = cv_results[0]
print(f"BEST: {best}")

# ---- skill_seed_bag_final_regressor: 3-seed bag on full train ----
seeds = [0, 1, 2]
test_preds_log = []
for s in seeds:
    pipe = make_pipeline(random_state=s, **best["params"])
    pipe.fit(X, y_log)
    p = pipe.predict(X_test)
    p = np.maximum(p, 0.0)
    test_preds_log.append(p)
final_log = np.mean(test_preds_log, axis=0)
final = np.expm1(final_log)
final = np.clip(final, 1.0, 30.0)  # target known in [1,29]; clip tiny

sub = pd.DataFrame({"id": test[ID].values, "Rings": final})
sub_path = os.path.join(WORK, "submission.csv")
sub.to_csv(sub_path, index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "RMSLE",
    "headline_value": best["mean_rmsle"],
    "all_metrics": {
        "cv_rmsle_best_mean": best["mean_rmsle"],
        "cv_rmsle_best_std": best["std_rmsle"],
        "cv_grid": cv_results,
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "HistGradientBoostingRegressor on log1p(Rings) (matches RMSLE). "
        "Tuned across a 3-config grid by 3-fold CV; refit best config with "
        "3 random_state seeds and averaged predictions in log space."
    ),
    "audit": {
        "hyperparameter_grid": grid,
        "chosen_config": best["params"],
        "seed_bag": seeds,
        "elapsed_sec": round(time.time() - t0, 2),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    },
}
with open(os.path.join(WORK, "result.json"), "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"Wrote {sub_path} (rows={len(sub)}) and result.json. Elapsed={time.time()-t0:.1f}s")
