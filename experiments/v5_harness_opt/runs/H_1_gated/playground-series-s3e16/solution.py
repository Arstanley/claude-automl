#!/usr/bin/env python3
"""Crab Age Prediction (playground-series-s3e16) — MAE regression.

Pipeline:
  - One-hot encode `Sex` (M/F/I), keep numeric features.
  - HistGradientBoostingRegressor with loss='absolute_error' to align with MAE.
  - 5-fold CV to pick a small explicit hyperparam config; report OOF MAE.
  - Seed-bag (3 seeds) for final refit on full train; average test predictions.

Skills applied:
  - skill_tune_when_n_train_supports_it.md
  - skill_seed_bag_final_regressor.md
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/"
    "playground-series-s3e16/data"
)
TARGET = "Age"
ID_COL = "id"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, sample


def prep(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    full = pd.concat([train[feature_cols], test[feature_cols]], axis=0)
    full = pd.get_dummies(full, columns=["Sex"], drop_first=False)
    X = full.iloc[: len(train)].reset_index(drop=True)
    X_test = full.iloc[len(train):].reset_index(drop=True)
    y = train[TARGET].astype(float).reset_index(drop=True)
    return X, y, X_test


def cv_mae(X: pd.DataFrame, y: pd.Series, params: dict, n_splits: int = 5) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr, va in kf.split(X):
        model = HistGradientBoostingRegressor(
            loss="absolute_error", random_state=0, **params
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(X.iloc[va])
    return mean_absolute_error(y, oof)


def main() -> None:
    t0 = time.time()
    train, test, sample = load()
    print(f"train={train.shape} test={test.shape} target={TARGET}")
    X, y, X_test = prep(train, test)

    # Small explicit grid (skill_tune_when_n_train_supports_it).
    grid = [
        {"max_iter": 400, "learning_rate": 0.05, "max_depth": None,
         "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 0.0,
         "early_stopping": False},
        {"max_iter": 600, "learning_rate": 0.05, "max_depth": None,
         "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 0.1,
         "early_stopping": False},
        {"max_iter": 800, "learning_rate": 0.03, "max_depth": None,
         "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0,
         "early_stopping": False},
    ]
    cv_results = []
    for i, params in enumerate(grid):
        score = cv_mae(X, y, params, n_splits=5)
        cv_results.append({"config": i, "params": params, "oof_mae": score})
        print(f"config {i}: oof_mae={score:.5f}")
    best = min(cv_results, key=lambda r: r["oof_mae"])
    best_params = best["params"]
    print(f"best config={best['config']} oof_mae={best['oof_mae']:.5f}")

    # Seed-bag (skill_seed_bag_final_regressor): refit on full train with 3 seeds.
    seeds = [0, 1, 2]
    test_preds = []
    for s in seeds:
        model = HistGradientBoostingRegressor(
            loss="absolute_error", random_state=s, **best_params
        )
        model.fit(X, y)
        test_preds.append(model.predict(X_test))
    final_pred = np.mean(np.vstack(test_preds), axis=0)
    # Clip to non-negative; crab Age is a positive integer count of rings.
    final_pred = np.clip(final_pred, 1.0, None)

    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: final_pred})
    out_csv = HERE / "submission.csv"
    sub.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} rows={len(sub)}")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "MAE",
        "headline_value": float(best["oof_mae"]),
        "all_metrics": {
            "oof_mae_best": float(best["oof_mae"]),
            "cv_results": [
                {"config": r["config"], "oof_mae": float(r["oof_mae"])}
                for r in cv_results
            ],
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_seed_bag_final_regressor.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor with loss='absolute_error' (matches MAE "
            "metric). Small 3-config grid via 5-fold CV picks best; final refit "
            "uses 3-seed bag averaged. Sex one-hot encoded; predictions clipped to "
            ">=1."
        ),
        "audit": {
            "seed_bag": seeds,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features_after_ohe": int(X.shape[1]),
            "best_params": best_params,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    }
    out_json = HERE / "result.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_json}")
    print(f"elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
