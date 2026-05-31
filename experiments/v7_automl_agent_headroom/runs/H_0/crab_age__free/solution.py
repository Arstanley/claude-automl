"""Crab Age regression — RMSLE.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=59240, GBM family, no size/latency tokens
  → small explicit grid over HistGradientBoostingRegressor.

Strategy: train on log1p(Age), one-hot Sex, small CV grid, refit on full train.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
if not DATA.exists():
    DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/crab_age/data")

TARGET = "Age"
ID = "id"
N_JOBS = 2
RNG = 42

t0 = time.time()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    y = train[TARGET].to_numpy()
    y_log = np.log1p(y)
    X = train.drop(columns=[TARGET, ID])
    X_test = test.drop(columns=[ID])

    cat_cols = ["Sex"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Small explicit grid — keep it tight within 3 min budget.
    grid = [
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.03, "max_iter": 1200, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"learning_rate": 0.08, "max_iter": 500, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 1.0},
    ]

    cv = KFold(n_splits=4, shuffle=True, random_state=RNG)

    best_rmsle = np.inf
    best_cfg = None
    best_oof = None
    cv_results = []

    for cfg in grid:
        model = Pipeline(
            [
                ("pre", pre),
                (
                    "gbm",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        random_state=RNG,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=30,
                        **cfg,
                    ),
                ),
            ]
        )

        # Manual OOF for true RMSLE on original scale.
        oof_log = np.zeros(len(y_log))
        for tr_idx, va_idx in cv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y_log[tr_idx]
            model.fit(X_tr, y_tr)
            oof_log[va_idx] = model.predict(X_va)

        oof_pred = np.expm1(oof_log)
        score = rmsle(y, oof_pred)
        cv_results.append({"cfg": cfg, "val_rmsle": score})
        print(f"cfg={cfg} val_rmsle={score:.5f} elapsed={time.time()-t0:.1f}s")
        if score < best_rmsle:
            best_rmsle = score
            best_cfg = cfg
            best_oof = oof_pred

    print(f"BEST cfg={best_cfg} val_rmsle={best_rmsle:.5f}")

    # Refit best on full train.
    final = Pipeline(
        [
            ("pre", pre),
            (
                "gbm",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    random_state=RNG,
                    early_stopping=False,
                    **best_cfg,
                ),
            ),
        ]
    )
    final.fit(X, y_log)
    test_pred = np.clip(np.expm1(final.predict(X_test)), 0, None)

    sub = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: test_pred})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "rmsle",
        "headline_value": best_rmsle,  # placeholder until test labels scored; harness will replace with test_rmsle
        "all_metrics": {
            "val_rmsle": best_rmsle,
            "cv_results": cv_results,
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "HistGradientBoostingRegressor on log1p(Age) with one-hot Sex. "
            "4-config grid CV'd with 4-fold KFold; best refit on full train. "
            "Log-target training aligns loss with RMSLE."
        ),
        "best_cfg": best_cfg,
        "elapsed_sec": time.time() - t0,
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"DONE elapsed={time.time()-t0:.1f}s val_rmsle={best_rmsle:.5f}")


if __name__ == "__main__":
    main()
