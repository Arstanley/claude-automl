"""Solution for playground-series-s3e6 (Paris Housing price RMSE).

Approach:
  1. Inspect data, find that a few training rows have implausible feature values
     (squareMeters > 1e5, floors > 100, made > 2025) while *all* test rows are
     within the plausible Paris-Housing range. Drop those outlier training rows
     so the model isn't pulled by synthetic noise that does not appear at test
     time.
  2. Use HistGradientBoostingRegressor (fast tree GBM in sklearn) with a small
     explicit hyperparameter grid evaluated via 5-fold KFold CV (RMSE).
  3. Refit the best config on the full cleaned train set and predict on test.

Outputs (in cwd):
  - submission.csv   id,price
  - result.json      headline metric + skill provenance
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e6/data"
)
OUT_DIR = Path(__file__).resolve().parent

TARGET = "price"
ID_COL = "id"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"train={train.shape} test={test.shape}")

    # Feature plausibility: test rows are all within these ranges.
    print(
        "test ranges:",
        {
            "squareMeters_max": int(test["squareMeters"].max()),
            "floors_max": int(test["floors"].max()),
            "made_max": int(test["made"].max()),
        },
    )

    # Drop noisy training rows whose feature values are far outside what the
    # test set ever exhibits (these are synthetic-generation artifacts).
    mask = (
        (train["squareMeters"] <= 100_000)
        & (train["floors"] <= 100)
        & (train["made"] <= 2025)
    )
    n_dropped = int((~mask).sum())
    clean = train.loc[mask].reset_index(drop=True)
    print(f"dropped {n_dropped} outlier rows ({n_dropped / len(train):.3%})")

    feats = [c for c in clean.columns if c not in (ID_COL, TARGET)]
    X = clean[feats]
    y = clean[TARGET]

    # ---- Hyperparameter grid for HistGradientBoosting (skill_tune_when_n_train_supports_it) ----
    # n_train ~ 18k, no size/latency constraint in the prompt, so a small grid is appropriate.
    grid = [
        dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20),
        dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20),
        dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10),
        dict(max_iter=1000, learning_rate=0.02, max_leaf_nodes=63, min_samples_leaf=20),
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for params in grid:
        fold_rmses = []
        for tr_idx, va_idx in cv.split(X):
            model = HistGradientBoostingRegressor(
                random_state=42, early_stopping=False, **params
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            fold_rmses.append(rmse(y.iloc[va_idx], model.predict(X.iloc[va_idx])))
        mean = float(np.mean(fold_rmses))
        std = float(np.std(fold_rmses))
        cv_results.append({"params": params, "cv_rmse_mean": mean, "cv_rmse_std": std})
        print(f"params={params} -> RMSE {mean:.1f} +- {std:.1f}")

    cv_results.sort(key=lambda r: r["cv_rmse_mean"])
    best = cv_results[0]
    print(f"\nbest config: {best}")

    # ---- Baseline: linear regression (sanity reference) ----
    lr_fold = []
    for tr_idx, va_idx in cv.split(X):
        m = LinearRegression().fit(X.iloc[tr_idx], y.iloc[tr_idx])
        lr_fold.append(rmse(y.iloc[va_idx], m.predict(X.iloc[va_idx])))
    lr_cv = float(np.mean(lr_fold))
    print(f"baseline LinearRegression CV RMSE: {lr_cv:.1f}")

    # ---- Refit best config on full cleaned train ----
    final = HistGradientBoostingRegressor(
        random_state=42, early_stopping=False, **best["params"]
    )
    final.fit(X, y)

    # ---- Predict on test ----
    X_test = test[feats]
    pred = final.predict(X_test)

    # Don't let predictions go negative (price >= 0).
    pred = np.clip(pred, 0.0, None)

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred})
    sub_path = OUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path} rows={len(sub)}")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "rmse",
        "headline_value": best["cv_rmse_mean"],
        "all_metrics": {
            "cv_rmse_mean": best["cv_rmse_mean"],
            "cv_rmse_std": best["cv_rmse_std"],
            "baseline_linear_cv_rmse": lr_cv,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_verify_split_claim_before_holdout.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor with a small 4-config grid (5-fold CV) on "
            f"Paris-Housing-style data. Dropped {n_dropped} training rows whose "
            "squareMeters / floors / made values fell outside the plausible range "
            "exhibited by the test set (synthetic-noise outliers). Best CV RMSE "
            f"{best['cv_rmse_mean']:.0f}; linear-regression baseline {lr_cv:.0f}."
        ),
        "audit": {
            "n_train_full": int(len(train)),
            "n_train_clean": int(len(clean)),
            "n_outliers_dropped": n_dropped,
            "hyperparameter_grid": [r["params"] for r in cv_results],
            "cv_results": cv_results,
            "chosen_config": best["params"],
            "features": feats,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    }

    res_path = OUT_DIR / "result.json"
    res_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {res_path}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
