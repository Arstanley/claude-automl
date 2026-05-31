"""Crab Age regression — RMSLE.

Skills applied:
- skill_tune_when_n_train_supports_it.md (n_train=59240 >> 500, no constraint tokens
  in prompt — small explicit grid over HistGradientBoostingRegressor configs, scored
  by RMSLE via 5-fold KFold CV).
- skill_stringify_low_cardinality_int_codes.md (Sex is already string M/F/I → just
  one-hot encode; verified no silent int code).

Train target log1p-transformed so MSE on log1p == RMSLE in original scale; predict,
expm1 back, clip at >=1 (Age min in train is 1) and round/cast to non-negative.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/crab_age/data")

N_JOBS = 2
RANDOM_STATE = 0
T0 = time.time()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def build_pipeline(**hgb_kwargs) -> Pipeline:
    cat_cols = ["Sex"]
    num_cols = [
        "Length",
        "Diameter",
        "Height",
        "Weight",
        "Shucked Weight",
        "Viscera Weight",
        "Shell Weight",
    ]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=RANDOM_STATE,
        early_stopping=False,
        **hgb_kwargs,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[t+{time.time()-T0:.1f}s] train={train.shape} test={test.shape}", flush=True)

    feat_cols = [c for c in train.columns if c not in ("id", "Age")]
    X = train[feat_cols]
    y = train["Age"].to_numpy()
    y_log = np.log1p(y)  # train on log1p so MSE in this space == RMSLE^2

    # Skill: small explicit grid (3 configs, no random search) — RMSLE via 5-fold CV.
    grid = [
        dict(max_iter=400, learning_rate=0.05, max_depth=6, l2_regularization=0.0),
        dict(max_iter=600, learning_rate=0.03, max_depth=7, l2_regularization=0.1),
        dict(max_iter=500, learning_rate=0.04, max_depth=None, max_leaf_nodes=63, l2_regularization=0.0),
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    def cv_score(params):
        scores = []
        for tr_idx, va_idx in cv.split(X):
            pipe = build_pipeline(**params)
            pipe.fit(X.iloc[tr_idx], y_log[tr_idx])
            preds = np.expm1(pipe.predict(X.iloc[va_idx]))
            scores.append(rmsle(y[va_idx], preds))
        return float(np.mean(scores)), float(np.std(scores))

    # Run CV serially (per-fold). Budget is tight (2 min); 5-fold * 3 configs on 59k rows
    # with HGB is ~30-60s.
    for params in grid:
        m, s = cv_score(params)
        cv_results.append({"params": params, "rmsle_mean": m, "rmsle_std": s})
        print(f"[t+{time.time()-T0:.1f}s]  {params} -> RMSLE {m:.5f} ± {s:.5f}", flush=True)

    cv_results.sort(key=lambda r: r["rmsle_mean"])
    best = cv_results[0]
    print(f"[t+{time.time()-T0:.1f}s] best={best['params']} cv_rmsle={best['rmsle_mean']:.5f}", flush=True)

    # Refit on full train, predict test.
    final = build_pipeline(**best["params"])
    final.fit(X, y_log)
    test_pred = np.expm1(final.predict(test[feat_cols]))
    test_pred = np.clip(test_pred, 1.0, None)  # train Age min is 1

    sub = pd.DataFrame({"id": test["id"], "Age": test_pred})
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"[t+{time.time()-T0:.1f}s] wrote submission.csv ({len(sub)} rows)", flush=True)

    result = {
        "target_column_chosen": "Age",
        "headline_metric": "RMSLE",
        "headline_value": best["rmsle_mean"],
        "all_metrics": {
            "cv_rmsle_mean": best["rmsle_mean"],
            "cv_rmsle_std": best["rmsle_std"],
            "cv_protocol": "KFold(5, shuffle=True, random_state=0) on log1p(Age) -> expm1 -> RMSLE",
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_stringify_low_cardinality_int_codes.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor on log1p(Age) (so squared-error loss equals "
            "RMSLE^2). Small 3-config grid (depth/lr/leaves) over 5-fold KFold; chose "
            "lowest mean RMSLE; refit on full train. Sex one-hot encoded; no NaNs in data."
        ),
        "audit": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "hyperparameter_grid": [r["params"] for r in cv_results],
            "cv_results": cv_results,
            "chosen_config": best["params"],
            "wall_seconds": time.time() - T0,
        },
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[t+{time.time()-T0:.1f}s] wrote result.json", flush=True)


if __name__ == "__main__":
    main()
