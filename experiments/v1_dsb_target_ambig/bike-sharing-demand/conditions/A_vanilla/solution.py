"""
Bike Sharing Demand — predict hourly `count`.

Approach
--------
- Target: `count` (total hourly rentals). The sample submission file expects
  `datetime,count`. Test rows do not contain `casual`/`registered`/`count`,
  so we cannot use those as features (they would be leakage and are
  unavailable at inference time).
- Metric: RMSLE. We train on log1p(count) with MSE loss so the model
  directly optimizes a proxy for RMSLE.
- Features engineered from `datetime`: year, month, day, dayofweek, hour.
  Categorical features: season, weather, holiday, workingday, plus the
  datetime-derived ones. Numeric: temp, atemp, humidity, windspeed.
- Model: GradientBoostingRegressor — a solid CPU-only baseline that
  handles non-linearities and interactions well for tabular ML.
- Validation: time-aware holdout using the last 4 days of each month from
  the training split (days 16-19) to mimic the train/test split scheme
  (train = first 19 days, test = days 20+).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = HERE / "submission.csv"
RESULT_PATH = HERE / "result.json"


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour
    return df


FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = add_datetime_features(train)
    test = add_datetime_features(test)

    X = train[FEATURES].values
    y = train["count"].values.astype(float)
    y_log = np.log1p(y)

    # Time-aware holdout: days 16-19 of each month
    holdout_mask = train["day"].between(16, 19).values
    X_tr, X_val = X[~holdout_mask], X[holdout_mask]
    y_tr_log, y_val = y_log[~holdout_mask], y[holdout_mask]

    val_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=0,
    )
    val_model.fit(X_tr, y_tr_log)
    val_pred = np.expm1(val_model.predict(X_val))
    val_pred = np.clip(val_pred, 0, None)
    val_rmsle = rmsle(y_val, val_pred)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    print(f"[validation] RMSLE={val_rmsle:.4f}  RMSE={val_rmse:.2f}  MAE={val_mae:.2f}")

    # Final model trained on all rows
    final_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=0,
    )
    final_model.fit(X, y_log)

    X_test = test[FEATURES].values
    test_pred = np.expm1(final_model.predict(X_test))
    test_pred = np.clip(test_pred, 0, None)

    submission = pd.DataFrame(
        {"datetime": test["datetime"].values, "count": test_pred}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH} ({len(submission)} rows)")

    # Training RMSLE on full training set (optimistic; for reference only)
    train_pred = np.expm1(final_model.predict(X))
    train_pred = np.clip(train_pred, 0, None)
    train_rmsle = rmsle(y, train_pred)

    result = {
        "target_column_chosen": "count",
        "headline_metric": "RMSLE",
        "headline_value": val_rmsle,
        "all_metrics": {
            "val_RMSLE": val_rmsle,
            "val_RMSE": val_rmse,
            "val_MAE": val_mae,
            "train_RMSLE_full_fit": train_rmsle,
            "n_train_rows": int(len(train)),
            "n_val_rows": int(holdout_mask.sum()),
            "n_test_rows": int(len(test)),
        },
        "interpretation_notes": (
            "Predicted `count` (total hourly rentals) since the sample "
            "submission expects `datetime,count` and test.csv lacks "
            "casual/registered/count (so they cannot be used as features). "
            "Model: GradientBoostingRegressor on log1p(count) with engineered "
            "datetime features (year, month, day, dayofweek, hour) plus the "
            "weather and calendar columns. Validation is a time-aware "
            "holdout using days 16-19 of each training month to mirror the "
            "train-vs-test day split. Headline RMSLE reflects that holdout."
        ),
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {RESULT_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
