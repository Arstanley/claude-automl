"""
Playground Series S3E5 — Wine Quality (ordinal classification, QWK metric).

Approach:
- Ordinal target {3..8}; QWK rewards getting the integer prediction close to the
  true bucket, so we treat this as a regression problem (HistGradientBoostingRegressor),
  then round + clip to the integer quality range. Regression-then-round is a
  widely-replicated winning trick for QWK on the wine-quality data.
- Small explicit hyperparameter grid (skill_tune_when_n_train_supports_it:
  n_train=1644, no size/latency constraint).
- Final prediction: average of n_seeds bagged regressors per chosen config, then
  round + clip to [3, 8].

Outputs:
- submission.csv : Id,quality
- result.json    : metrics + audit
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e5/data"
)
HERE = Path(__file__).resolve().parent

TARGET = "quality"
ID_COL = "Id"
QMIN, QMAX = 3, 8  # observed integer range of training quality


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def to_int_preds(y_float):
    return np.clip(np.round(y_float).astype(int), QMIN, QMAX)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X = train[feature_cols].values.astype(np.float32)
    y = train[TARGET].values.astype(np.int64)
    X_test = test[feature_cols].values.astype(np.float32)

    # --- small explicit grid (skill_tune_when_n_train_supports_it) ---
    grid = [
        dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
        dict(max_iter=500, learning_rate=0.05, max_depth=None, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=0.0),
        dict(max_iter=800, learning_rate=0.02, max_depth=None, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for cfg_idx, params in enumerate(grid):
        fold_qwks = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            model = HistGradientBoostingRegressor(
                **params,
                early_stopping=False,
                random_state=0,
            )
            model.fit(X[tr_idx], y[tr_idx].astype(np.float32))
            pred_float = model.predict(X[va_idx])
            pred_int = to_int_preds(pred_float)
            fold_qwks.append(qwk(y[va_idx], pred_int))
        mean = float(np.mean(fold_qwks))
        std = float(np.std(fold_qwks))
        cv_results.append({"config": params, "mean_qwk": mean, "std_qwk": std})
        print(f"cfg {cfg_idx}: QWK = {mean:.4f} +/- {std:.4f}  params={params}")

    cv_results.sort(key=lambda r: r["mean_qwk"], reverse=True)
    best = cv_results[0]
    best_params = best["config"]
    print(f"\nBest config: QWK {best['mean_qwk']:.4f} +/- {best['std_qwk']:.4f}")
    print(f"Params: {best_params}")

    # --- bagged refit on full train: 5 seeds, average regression output ---
    seeds = [0, 1, 2, 3, 4]
    test_pred_float = np.zeros(len(test), dtype=np.float64)
    for s in seeds:
        m = HistGradientBoostingRegressor(
            **best_params,
            early_stopping=False,
            random_state=s,
        )
        m.fit(X, y.astype(np.float32))
        test_pred_float += m.predict(X_test)
    test_pred_float /= len(seeds)
    test_pred_int = to_int_preds(test_pred_float)

    # --- submission ---
    submission = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred_int})
    submission.to_csv(HERE / "submission.csv", index=False)
    print(f"\nWrote submission.csv with {len(submission)} rows")
    print("Predicted quality distribution:")
    print(pd.Series(test_pred_int).value_counts().sort_index().to_dict())

    # --- also compute training-fold-bagged OOF QWK for the chosen config as audit ---
    oof_float = np.zeros(len(train), dtype=np.float64)
    for s in seeds:
        skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + s)
        oof_s = np.zeros(len(train), dtype=np.float64)
        for tr_idx, va_idx in skf_s.split(X, y):
            m = HistGradientBoostingRegressor(
                **best_params,
                early_stopping=False,
                random_state=s,
            )
            m.fit(X[tr_idx], y[tr_idx].astype(np.float32))
            oof_s[va_idx] = m.predict(X[va_idx])
        oof_float += oof_s
    oof_float /= len(seeds)
    oof_int = to_int_preds(oof_float)
    oof_qwk = float(qwk(y, oof_int))
    print(f"OOF (bagged) QWK on chosen config: {oof_qwk:.4f}")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "quadratic_weighted_kappa",
        "headline_value": oof_qwk,
        "all_metrics": {
            "cv_qwk_mean_best": best["mean_qwk"],
            "cv_qwk_std_best": best["std_qwk"],
            "oof_bagged_qwk": oof_qwk,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Ordinal target {3..8} with QWK; modeled as regression with "
            "HistGradientBoostingRegressor then round+clip to int range (a "
            "well-known QWK-friendly trick on wine quality). Ran a 4-config "
            "stratified-5-fold tune and bagged 5 seeds on the winner for "
            "lower-variance test predictions."
        ),
        "audit": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "features": feature_cols,
            "hyperparameter_grid": [r["config"] for r in cv_results],
            "cv_results": cv_results,
            "chosen_config": best_params,
            "n_bag_seeds": len(seeds),
            "wall_seconds": round(time.time() - t0, 2),
        },
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nDone in {time.time() - t0:.1f}s. Wrote result.json")


if __name__ == "__main__":
    main()
