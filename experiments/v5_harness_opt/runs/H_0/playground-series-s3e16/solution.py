"""
Solution for playground-series-s3e16: Crab Age Prediction.
Metric: MAE. Target: Age (int >= 1).

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=59240, GBM family, no
  size/latency constraint -> use a small 2-config grid with KFold CV.

Submission format per spec: header `id,Age`, one row per test id.
(sample_submission.csv in DATA_DIR has stale IDs that don't match
test.csv; we follow the actual test ids per the spec.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/"
    "playground-series-s3e16/data"
)

TARGET = "Age"
ID = "id"

CONFIGS = [
    # 2-config small grid (budget: <= 2 configs)
    {"max_iter": 400, "learning_rate": 0.05, "max_depth": 7,
     "min_samples_leaf": 30, "l2_regularization": 0.0},
    {"max_iter": 600, "learning_rate": 0.05, "max_depth": 8,
     "min_samples_leaf": 20, "l2_regularization": 1.0},
]


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # One-hot Sex (3 levels: M/F/I). HistGBR handles categoricals via
    # categorical_features but plain OHE is fine for this n.
    df = pd.get_dummies(df, columns=["Sex"], drop_first=False)
    # Light interaction features that frequently help on this dataset.
    df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["MeatRatio"] = df["Shucked Weight"] / (df["Weight"] + 1e-9)
    df["ShellRatio"] = df["Shell Weight"] / (df["Weight"] + 1e-9)
    return df


def main() -> None:
    t0 = time.time()
    train, test = load()
    y = train[TARGET].astype(float).values
    test_ids = test[ID].values

    X_train_full = featurize(train.drop(columns=[ID, TARGET]))
    X_test = featurize(test.drop(columns=[ID]))
    # Align columns in case OHE produced different sets.
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    test_preds_by_cfg = []

    for cfg_idx, cfg in enumerate(CONFIGS):
        oof = np.zeros(len(y))
        test_pred = np.zeros(len(test))
        for fold, (tr, va) in enumerate(kf.split(X_train_full)):
            model = HistGradientBoostingRegressor(
                loss="absolute_error",  # match MAE metric
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                **cfg,
            )
            model.fit(X_train_full.iloc[tr], y[tr])
            oof[va] = model.predict(X_train_full.iloc[va])
            test_pred += model.predict(X_test) / kf.n_splits
        mae = mean_absolute_error(y, oof)
        cv_results.append({"cfg_idx": cfg_idx, "cfg": cfg, "cv_mae": mae})
        test_preds_by_cfg.append(test_pred)
        print(f"[cfg{cfg_idx}] CV MAE = {mae:.5f} | cfg={cfg}")

    best = min(range(len(CONFIGS)), key=lambda i: cv_results[i]["cv_mae"])
    best_mae = cv_results[best]["cv_mae"]
    test_pred = test_preds_by_cfg[best]
    # Age is a positive integer; clip to observed range.
    test_pred = np.clip(test_pred, 1.0, 30.0)

    sub = pd.DataFrame({ID: test_ids, TARGET: test_pred})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "MAE",
        "headline_value": float(best_mae),
        "all_metrics": {
            "cv_mae_by_config": [
                {"cfg_idx": r["cfg_idx"], "cv_mae": float(r["cv_mae"]),
                 "cfg": r["cfg"]}
                for r in cv_results
            ],
            "best_cfg_idx": best,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "wall_seconds": round(time.time() - t0, 2),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor with loss=absolute_error (matches "
            "MAE), 5-fold CV, small 2-config grid per harness budget. "
            "Light feature engineering (Volume, MeatRatio, ShellRatio) on top "
            "of OHE Sex. Used test.csv ids for submission since "
            "sample_submission.csv's ids don't match the actual test set."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best cfg={best} CV MAE={best_mae:.5f}")
    print(f"Wrote submission.csv ({len(sub)} rows) and result.json")


if __name__ == "__main__":
    main()
