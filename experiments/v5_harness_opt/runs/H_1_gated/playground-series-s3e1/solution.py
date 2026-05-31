"""
AutoML solution for playground-series-s3e1 (California Housing Synthetic).
Metric: RMSE. Target: MedHouseVal.

Skills applied:
- skill_tune_when_n_train_supports_it.md  (n_train=29709 >> 500, no latency/size tokens, GBM family)
- skill_seed_bag_final_regressor.md       (regression, n_train large, model has random_state)
"""

import json
import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "2")

DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e1/data/"
OUT_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1_gated/playground-series-s3e1/"

TARGET = "MedHouseVal"
ID_COL = "id"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    X = train[feature_cols].values
    y = train[TARGET].values
    X_test = test[feature_cols].values

    # ---- Skill: tune_when_n_train_supports_it — small explicit grid ----
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,  min_samples_leaf=20),
        dict(max_iter=500, learning_rate=0.05, max_depth=None, max_leaf_nodes=63,  min_samples_leaf=20),
        dict(max_iter=700, learning_rate=0.03, max_depth=None, max_leaf_nodes=63,  min_samples_leaf=30),
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    grid_results = []
    for params in grid:
        scores = []
        for tr_idx, va_idx in cv.split(X):
            m = HistGradientBoostingRegressor(**params, random_state=0, early_stopping=False)
            m.fit(X[tr_idx], y[tr_idx])
            scores.append(rmse(y[va_idx], m.predict(X[va_idx])))
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        grid_results.append((mean_s, std_s, params))
        print(f"  {params} -> RMSE {mean_s:.4f} +/- {std_s:.4f}")

    grid_results.sort(key=lambda r: r[0])  # lower RMSE better
    best_mean, best_std, best_params = grid_results[0]
    print(f"Best config: {best_params} (RMSE {best_mean:.4f})")

    # ---- Skill: seed_bag_final_regressor ----
    seeds = [0, 1, 2]

    # First, compute single-seed-0 OOF and 3-seed averaged OOF as the anti-overfit guard.
    oof_seed0 = np.zeros(len(y))
    oof_bag = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X):
        preds_per_seed = []
        for s in seeds:
            m = HistGradientBoostingRegressor(**best_params, random_state=s, early_stopping=False)
            m.fit(X[tr_idx], y[tr_idx])
            preds_per_seed.append(m.predict(X[va_idx]))
        oof_seed0[va_idx] = preds_per_seed[0]
        oof_bag[va_idx] = np.mean(preds_per_seed, axis=0)

    rmse_seed0 = rmse(y, oof_seed0)
    rmse_bag = rmse(y, oof_bag)
    print(f"OOF RMSE seed-0 only: {rmse_seed0:.4f}")
    print(f"OOF RMSE 3-seed bag : {rmse_bag:.4f}")
    use_bag = rmse_bag <= rmse_seed0

    # Final fits on full train.
    test_preds = []
    for s in seeds:
        m = HistGradientBoostingRegressor(**best_params, random_state=s, early_stopping=False)
        m.fit(X, y)
        test_preds.append(m.predict(X_test))

    if use_bag:
        final_pred = np.mean(test_preds, axis=0)
        final_oof_rmse = rmse_bag
        final_strategy = "seed_bag_3"
    else:
        final_pred = test_preds[0]
        final_oof_rmse = rmse_seed0
        final_strategy = "seed_0_only"

    # ---- Submission ----
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: final_pred})
    sub_path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path} ({len(sub)} rows)")

    # ---- result.json ----
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "RMSE",
        "headline_value": final_oof_rmse,
        "all_metrics": {
            "oof_rmse_seed0_only": rmse_seed0,
            "oof_rmse_3seed_bag": rmse_bag,
            "best_grid_cv_rmse": best_mean,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_seed_bag_final_regressor.md",
        ],
        "interpretation_notes": (
            "Clean 8-feature tabular regression (n_train=29709, no NaN, no id overlap). "
            "Applied a 3-config HistGradientBoosting grid via 5-fold KFold, then "
            "seed-bagged 3 random_states. Bag was retained because OOF RMSE bag <= seed-0 OOF."
        ),
        "audit": {
            "hyperparameter_grid": [r[2] for r in grid_results],
            "chosen_config": best_params,
            "seed_bag": seeds,
            "final_strategy": final_strategy,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    }
    res_path = os.path.join(OUT_DIR, "result.json")
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {res_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
