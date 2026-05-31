"""
Tabular Playground Series Jul 2021 — multi-target air-pollution regression.

Targets: target_carbon_monoxide, target_benzene, target_nitrogen_oxides.
Metric:  mean column-wise RMSLE.

Strategy
--------
- 5688 train rows / 1423 test rows; train/test interleave temporally (verified empirically),
  so a plain KFold is appropriate (skill_verify_split_claim_before_holdout).
- All targets are strictly positive and skewed -> train models on log1p(y) and invert
  with expm1 so MSE under log-space corresponds to RMSLE in original space.
- HistGradientBoostingRegressor: handles non-linear feature interactions, fast, no GPU needed.
- Small explicit tuning grid (3 configs) per target via 3-fold CV
  (skill_tune_when_n_train_supports_it: n>=500, GBM, no latency/size constraints).
- Add cyclic/calendar features from the timestamp (hour, dayofweek, month, day_of_year sin/cos).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/tabular-playground-series-jul-2021/data")
WORKING_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_0/tabular-playground-series-jul-2021")

TARGETS = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
SEED = 42
N_JOBS = 2


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["date_time"])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    doy = dt.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def main() -> None:
    t_start = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train = add_time_features(train)
    test = add_time_features(test)

    feature_cols = [
        "deg_C", "relative_humidity", "absolute_humidity",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "hour", "dayofweek", "month", "day",
        "doy_sin", "doy_cos", "hour_sin", "hour_cos",
    ]

    X = train[feature_cols].to_numpy()
    X_test = test[feature_cols].to_numpy()

    # 3-config tuning grid (skill_tune_when_n_train_supports_it).
    param_grid = [
        {"max_iter": 400, "learning_rate": 0.06, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"max_iter": 600, "learning_rate": 0.05, "max_depth": 7, "min_samples_leaf": 30, "l2_regularization": 0.1},
        {"max_iter": 800, "learning_rate": 0.04, "max_depth": 8, "min_samples_leaf": 40, "l2_regularization": 0.5},
    ]

    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)

    per_target_cv = {}
    per_target_best = {}
    test_preds = {}

    for target in TARGETS:
        y = train[target].to_numpy()
        y_log = np.log1p(y)

        best_cfg = None
        best_score = np.inf
        cfg_scores = []
        for cfg in param_grid:
            fold_scores = []
            for tr_idx, va_idx in kf.split(X):
                model = HistGradientBoostingRegressor(
                    random_state=SEED,
                    **cfg,
                )
                model.fit(X[tr_idx], y_log[tr_idx])
                pred_log = model.predict(X[va_idx])
                pred = np.expm1(pred_log)
                fold_scores.append(rmsle(y[va_idx], pred))
            mean_score = float(np.mean(fold_scores))
            cfg_scores.append({"cfg": cfg, "rmsle": mean_score})
            if mean_score < best_score:
                best_score = mean_score
                best_cfg = cfg

        per_target_cv[target] = {"best_rmsle": best_score, "grid": cfg_scores}
        per_target_best[target] = best_cfg

        # Refit on full train with best config
        final_model = HistGradientBoostingRegressor(random_state=SEED, **best_cfg)
        final_model.fit(X, y_log)
        pred_test = np.expm1(final_model.predict(X_test))
        pred_test = np.clip(pred_test, a_min=0, a_max=None)
        test_preds[target] = pred_test

    headline = float(np.mean([per_target_cv[t]["best_rmsle"] for t in TARGETS]))

    submission = pd.DataFrame({"date_time": test["date_time"].values})
    for target in TARGETS:
        submission[target] = test_preds[target]
    submission.to_csv(WORKING_DIR / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGETS,
        "headline_metric": "mean_column_RMSLE",
        "headline_value": headline,
        "all_metrics": {
            "per_target_best_rmsle": {t: per_target_cv[t]["best_rmsle"] for t in TARGETS},
            "per_target_best_cfg": {t: per_target_best[t] for t in TARGETS},
        },
        "skills_applied": [
            "skill_verify_split_claim_before_holdout.md",
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Three positive-valued regression targets evaluated by RMSLE. "
            "Trained HistGradientBoosting on log1p(y) so MSE in log-space matches RMSLE; "
            "selected from a 3-config grid via 3-fold KFold CV (train/test interleave temporally, "
            "verified by overlapping date ranges, so random KFold is valid). Calendar/cyclic "
            "features added from date_time."
        ),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(WORKING_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps({"headline": headline, "per_target": {t: per_target_cv[t]["best_rmsle"] for t in TARGETS}, "runtime_s": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
