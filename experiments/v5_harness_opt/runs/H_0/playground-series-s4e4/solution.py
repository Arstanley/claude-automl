"""
Playground Series S4E4 — Abalone Rings regression (RMSLE).

Strategy
--------
- 72k train rows, 9 features, single low-cardinality categorical ("Sex"),
  no missing values. Classic small-tabular regression.
- Metric is RMSLE -> train on log1p(Rings), predict, then expm1.
  Under log target, MSE ~= RMSLE^2 on the original-scale target, so an
  ordinary squared-loss GBM is well-aligned with the metric.
- HistGradientBoostingRegressor is fast (~tens of seconds for 72k rows)
  and competitive on tabular data; no third-party deps.
- Small 3-config tuning grid (skill_tune_when_n_train_supports_it).
- One-hot encode "Sex" (3 levels) — HGBR also natively handles categoricals,
  but OHE keeps the pipeline explicit and is essentially free at this size.
- 5-fold CV on log-target with squared-error -> back-transform predictions
  and compute true RMSLE on the original scale for the headline metric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e4/data")
OUT_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42
N_JOBS = 2


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def make_pipeline(params: dict) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
        ],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=RANDOM_STATE,
        early_stopping=False,
        **params,
    )
    return Pipeline([("pre", pre), ("model", model)])


def cv_score(pipe: Pipeline, X: pd.DataFrame, y_log: np.ndarray, y_true: np.ndarray) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))
    for fold, (tr, va) in enumerate(kf.split(X)):
        pipe.fit(X.iloc[tr], y_log[tr])
        oof[va] = pipe.predict(X.iloc[va])
    preds = np.expm1(oof)
    return rmsle(y_true, preds)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target = "Rings"
    feature_cols = [c for c in train.columns if c not in ("id", target)]
    X = train[feature_cols].copy()
    y = train[target].astype(float).values
    y_log = np.log1p(y)
    X_test = test[feature_cols].copy()

    # 3-config grid — vary capacity along a single axis.
    grid = [
        {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"max_iter": 800, "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 0.1},
    ]

    results = []
    for params in grid:
        pipe = make_pipeline(params)
        score = cv_score(pipe, X, y_log, y)
        results.append({"params": params, "cv_rmsle": score})
        print(f"params={params} -> CV RMSLE={score:.5f}")

    best = min(results, key=lambda r: r["cv_rmsle"])
    print(f"Best CV RMSLE: {best['cv_rmsle']:.5f} with params={best['params']}")

    # Refit on full train with best config.
    final = make_pipeline(best["params"])
    final.fit(X, y_log)
    test_pred = np.expm1(final.predict(X_test))
    test_pred = np.clip(test_pred, 1, None)  # Rings has min=1

    sub = pd.DataFrame({"id": test["id"].values, "Rings": test_pred})
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    result = {
        "target_column_chosen": target,
        "headline_metric": "RMSLE",
        "headline_value": best["cv_rmsle"],
        "all_metrics": {
            "cv_rmsle_best": best["cv_rmsle"],
            "cv_rmsle_all": [r["cv_rmsle"] for r in results],
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Trained HistGradientBoostingRegressor on log1p(Rings) so squared-error "
            "loss aligns with the RMSLE metric. 5-fold CV picked the best of a "
            "3-config capacity grid; refit on full train and predicted on test."
        ),
        "best_params": best["params"],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
