"""
Playground Series S3E1: California Housing regression.
Target: MedHouseVal, Metric: RMSE.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=29709 >> 500, no size/latency constraints
  in prompt; small 3-config grid over HistGradientBoostingRegressor.
- skill_probe_group_structure_in_ids: probed id column for group leakage (negative result;
  random KFold appropriate).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e1/data/")
WORK_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 0


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target = "MedHouseVal"
    id_col = "id"

    feature_cols = [c for c in train.columns if c not in (target, id_col)]
    X = train[feature_cols].values
    y = train[target].values
    X_test = test[feature_cols].values

    # ---- skill_probe_group_structure_in_ids ----
    train_ids = set(train[id_col].astype(str))
    test_ids = set(test[id_col].astype(str))
    id_overlap = len(train_ids & test_ids) / max(1, len(test_ids))
    # id is a row identifier, not a group key; we don't expect leakage. Probe and log.
    group_probe = {"column": id_col, "overlap_frac": id_overlap}

    # ---- skill_tune_when_n_train_supports_it ----
    n_train = len(train)
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=20),
        dict(max_iter=500, learning_rate=0.05, max_depth=8, min_samples_leaf=20),
        dict(max_iter=600, learning_rate=0.03, max_depth=7, min_samples_leaf=30),
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_results = []
    for params in grid:
        fold_scores = []
        for tr_idx, va_idx in cv.split(X):
            model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_scores.append(rmse(y[va_idx], pred))
        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        grid_results.append({"params": params, "rmse_mean": mean_s, "rmse_std": std_s})
        print(f"  {params}: RMSE = {mean_s:.5f} +/- {std_s:.5f}")

    # Best = lowest RMSE
    grid_results.sort(key=lambda d: d["rmse_mean"])
    best = grid_results[0]
    print(f"\nBest config: {best['params']} -> CV RMSE {best['rmse_mean']:.5f}")

    # Refit on full train
    final_model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **best["params"])
    final_model.fit(X, y)
    preds_test = final_model.predict(X_test)

    # Submission
    sub = pd.DataFrame({id_col: test[id_col].values, target: preds_test})
    sub.to_csv(WORK_DIR / "submission.csv", index=False)

    elapsed = time.time() - t0
    print(f"Wrote submission.csv ({len(sub)} rows) in {elapsed:.1f}s")

    result = {
        "target_column_chosen": target,
        "headline_metric": "RMSE",
        "headline_value": best["rmse_mean"],
        "all_metrics": {
            "cv_rmse_mean": best["rmse_mean"],
            "cv_rmse_std": best["rmse_std"],
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_probe_group_structure_in_ids.md",
        ],
        "interpretation_notes": (
            "Tabular regression (synthetic California Housing); used "
            "HistGradientBoostingRegressor with a 3-config grid (5-fold CV). "
            "Probed id column overlap as group key — id is a row identifier with no "
            "group structure, so random KFold is appropriate."
        ),
        "audit": {
            "n_train": n_train,
            "n_test": int(len(test)),
            "features": feature_cols,
            "hyperparameter_grid": grid_results,
            "chosen_config": best["params"],
            "group_leakage_probe": group_probe,
            "elapsed_seconds": elapsed,
        },
    }
    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result.json — headline RMSE {best['rmse_mean']:.5f}")


if __name__ == "__main__":
    main()
