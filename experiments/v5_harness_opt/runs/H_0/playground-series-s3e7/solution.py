"""
Solution for playground-series-s3e7 (reservation cancellation - binary, ROC AUC).

Approach:
- Binary classification with HistGradientBoostingClassifier (fast, handles categoricals
  natively and tabular GBMs are SOTA on this kind of synthetic Kaggle tabular data).
- Light feature engineering: combine arrival_year/month/date into a numeric date proxy
  + interactions; treat low-cardinality int codes as categorical.
- Applied skill: skill_tune_when_n_train_supports_it (n_train=33680 >> 500, GBM family,
  no size/latency/production tokens in prompt). Small explicit grid (3 configs) tuned
  via 5-fold StratifiedKFold OOF ROC AUC.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e7/data")
HERE = Path(__file__).resolve().parent

RNG = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Build a numeric arrival-date ordinal (days since 2017-01-01). Some rows may have
    # an invalid (month, date) combo (e.g. Feb 30) — coerce to NaT then to a sentinel.
    dates = pd.to_datetime(
        dict(year=df["arrival_year"], month=df["arrival_month"], day=df["arrival_date"]),
        errors="coerce",
    )
    base = pd.Timestamp("2017-01-01")
    df["arrival_ord"] = (dates - base).dt.days.astype("float64")
    df["arrival_dow"] = dates.dt.dayofweek.astype("float64")
    df["arrival_quarter"] = dates.dt.quarter.astype("float64")

    # Composite/derived features
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["price_per_guest"] = df["avg_price_per_room"] / np.maximum(df["total_guests"], 1)
    df["weekend_ratio"] = df["no_of_weekend_nights"] / np.maximum(df["total_nights"], 1)
    df["prev_total"] = df["no_of_previous_cancellations"] + df["no_of_previous_bookings_not_canceled"]
    df["prev_cancel_rate"] = df["no_of_previous_cancellations"] / np.maximum(df["prev_total"], 1)
    return df


CAT_COLS = [
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "market_segment_type",
    "repeated_guest",
    "arrival_year",
    "arrival_month",
    "arrival_date",
]


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    y = train["booking_status"].values
    train_feat = build_features(train.drop(columns=["booking_status"]))
    test_feat = build_features(test)

    feat_cols = [c for c in train_feat.columns if c != "id"]
    cat_mask = [c in CAT_COLS for c in feat_cols]

    X = train_feat[feat_cols]
    X_test = test_feat[feat_cols]

    # Small explicit grid (3 configs) — skill_tune_when_n_train_supports_it.
    configs = [
        dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=800, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.1),
        dict(learning_rate=0.03, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.0),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    cv_results = []

    for cfg_idx, cfg in enumerate(configs):
        oof = np.zeros(len(X))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            model = HistGradientBoostingClassifier(
                **cfg,
                categorical_features=cat_mask,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=RNG + fold,
            )
            model.fit(X.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        auc = roc_auc_score(y, oof)
        cv_results.append({"config": cfg, "cv_auc": float(auc)})
        print(f"[cfg {cfg_idx}] AUC={auc:.5f}  cfg={cfg}")

    # Pick the best config and refit on full train (with early-stopping retained).
    best = max(cv_results, key=lambda r: r["cv_auc"])
    print(f"Best CV AUC = {best['cv_auc']:.5f} with cfg {best['config']}")

    # Build a small ensemble: refit the best cfg on full data several times with
    # different seeds and average. Cheap and slightly more stable.
    test_preds = np.zeros(len(X_test))
    n_seeds = 3
    for s in range(n_seeds):
        model = HistGradientBoostingClassifier(
            **best["config"],
            categorical_features=cat_mask,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=RNG + 100 * s,
        )
        model.fit(X, y)
        test_preds += model.predict_proba(X_test)[:, 1] / n_seeds

    sub = pd.DataFrame({"id": test["id"].values, "booking_status": test_preds})
    # sample_submission has float probabilities — the metric is AUC so probs are fine.
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)

    result = {
        "target_column_chosen": "booking_status",
        "headline_metric": "roc_auc",
        "headline_value": float(best["cv_auc"]),
        "all_metrics": {
            "cv_auc_per_config": [r["cv_auc"] for r in cv_results],
            "best_cv_auc": float(best["cv_auc"]),
            "n_train": int(len(X)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feat_cols)),
            "n_categorical": int(sum(cat_mask)),
            "n_seeds_ensemble": n_seeds,
            "wall_time_sec": float(time.time() - t0),
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "Binary classification with HistGradientBoostingClassifier and native categorical "
            "handling on engineered features (arrival ordinal/dow, totals, ratios). Applied "
            "tune_when_n_train_supports_it: n=33680, GBM, no constraints — used a 3-config grid "
            "with 5-fold StratifiedKFold OOF AUC for selection; refit best config with 3-seed "
            "ensemble for the test predictions."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {sub_path} and result.json. Wall time = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
