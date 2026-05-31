"""
Playground Series S3E1 — California housing synthetic, regression on MedHouseVal.
Metric: RMSE.

Skill evaluation summary (mechanical preconditions over INDEX.md):
 1) probe_group_structure_in_ids: feature overlap is from continuous numerics (lat/lon bins),
    no compound IDs, no leakable group key -> NOT APPLIED.
 2) train_on_listed_auxiliary_files: prompt mentions original California Housing as "feel free to
    use" but no auxiliary file is shipped in DATA_DIR -> NOT APPLIED.
 3) verify_split_claim_before_holdout: prompt makes no structural train/test split claim ->
    NOT APPLIED.
 4) stringify_low_cardinality_int_codes: no int-coded low-card columns; all features float ->
    NOT APPLIED.
 5) multi_axis_constraint_budgets: only one optimization axis (RMSE) + a training-time budget ->
    NOT APPLIED.
 6) missing_indicator_when_missingness_signals_class: zero NaNs in train/test -> NOT APPLIED.
 7) tune_when_n_train_supports_it: n_train=29709 >= 500, GBM family, but the harness budget says
    "training under 2 min" (an explicit time token) — APPLIED in restrained form: a tiny 3-config
    sweep over learning_rate/max_iter for HistGradientBoostingRegressor under a strict wall clock.
 8) tune_decision_threshold_for_f1: not classification -> NOT APPLIED.
 9) log_target_for_skewed_nonnegative_regression: y in [0.15, 5.0], skew=0.965 (<1.0),
    max/median=2.76 (<20), metric=RMSE (not RMSLE) -> NOT APPLIED.
10) blend_two_model_families: tabular and n>=1000, but the "training under 2 min" token
    explicitly fences out the extra-family budget -> NOT APPLIED.

Feature engineering: simple, cheap geospatial cluster id (KMeans on lat/lon) and a few ratios.
Model: HistGradientBoostingRegressor (CPU, fast, handles non-linear interactions well).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e1/data")
N_JOBS = 2  # harness directive
SEED = 42

t0 = time.time()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_household"] = df["AveRooms"]
    df["bedrooms_per_room"] = df["AveBedrms"] / (df["AveRooms"] + 1e-9)
    df["pop_per_house"] = df["Population"] / (df["AveOccup"] + 1e-9)
    df["log_pop"] = np.log1p(df["Population"])
    df["lat_lon"] = df["Latitude"] * df["Longitude"]
    return df


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    y = train["MedHouseVal"].values
    test_ids = test["id"].values

    base_cols = [c for c in train.columns if c not in ("id", "MedHouseVal")]
    X_tr_raw = train[base_cols].copy()
    X_te_raw = test[base_cols].copy()

    # Geospatial cluster id from lat/lon — cheap, often helps tree models on this data.
    km = MiniBatchKMeans(n_clusters=24, random_state=SEED, n_init="auto", batch_size=2048)
    km.fit(X_tr_raw[["Latitude", "Longitude"]].values)
    X_tr_raw["geo_cluster"] = km.predict(X_tr_raw[["Latitude", "Longitude"]].values)
    X_te_raw["geo_cluster"] = km.predict(X_te_raw[["Latitude", "Longitude"]].values)

    X_tr = add_features(X_tr_raw)
    X_te = add_features(X_te_raw)

    feat_cols = list(X_tr.columns)
    print(f"[{time.time() - t0:5.1f}s] features ready: {feat_cols}")

    # ---- light 3-config sweep, 5-fold OOF (early stopping keeps it fast) ----
    configs = [
        dict(learning_rate=0.07, max_iter=500, max_leaf_nodes=63, min_samples_leaf=40,
             l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20,
             l2_regularization=0.5),
        dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=30,
             l2_regularization=0.1),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = []
    best = None  # (rmse, cfg_idx, oof, test_preds)

    for ci, cfg in enumerate(configs):
        oof = np.zeros(len(X_tr))
        test_acc = np.zeros(len(X_te))
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=SEED + fold,
                **cfg,
            )
            model.fit(X_tr.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = model.predict(X_tr.iloc[va_idx])
            test_acc += model.predict(X_te) / kf.n_splits
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        cv_results.append({"cfg_idx": ci, "config": cfg, "rmse": rmse})
        print(f"[{time.time() - t0:5.1f}s] cfg {ci} RMSE={rmse:.5f}")
        if best is None or rmse < best[0]:
            best = (rmse, ci, oof, test_acc)

    best_rmse, best_idx, best_oof, best_test = best
    print(f"[{time.time() - t0:5.1f}s] best cfg {best_idx} RMSE={best_rmse:.5f}")

    # Clip predictions to observed y range (target is bounded by the original California dataset)
    y_lo, y_hi = float(y.min()), float(y.max())
    preds = np.clip(best_test, y_lo, y_hi)

    # ---- write submission ----
    sub = pd.DataFrame({"id": test_ids, "MedHouseVal": preds})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[{time.time() - t0:5.1f}s] wrote {sub_path}  rows={len(sub)}")

    # ---- result.json ----
    result = {
        "target_column_chosen": "MedHouseVal",
        "headline_metric": "rmse",
        "headline_value": best_rmse,
        "all_metrics": {
            "rmse_oof_best": best_rmse,
            "rmse_per_config": {str(r["cfg_idx"]): r["rmse"] for r in cv_results},
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "skills_evaluated_not_applied": {
            "skill_probe_group_structure_in_ids.md":
                "No compound IDs; train/test overlap is from continuous numerics (lat/lon bins), not groupable keys.",
            "skill_train_on_listed_auxiliary_files.md":
                "Prompt mentions original California Housing dataset is allowed but no auxiliary file is shipped in DATA_DIR.",
            "skill_verify_split_claim_before_holdout.md":
                "Prompt makes no structural train/test split claim to verify.",
            "skill_stringify_low_cardinality_int_codes.md":
                "All feature columns are float; no int-coded low-cardinality categoricals.",
            "skill_multi_axis_constraint_budgets.md":
                "Only one optimization axis (RMSE) plus a single training-time budget.",
            "skill_missing_indicator_when_missingness_signals_class.md":
                "Zero NaNs in train and test.",
            "skill_tune_decision_threshold_for_f1.md":
                "Regression task, not classification with F-style metric.",
            "skill_log_target_for_skewed_nonnegative_regression.md":
                "y skew=0.97 (<1.0), max/median=2.76 (<20), metric=RMSE (not RMSLE); precondition fails.",
            "skill_blend_two_model_families.md":
                "Harness budget explicitly caps training under 2 min (latency/budget token), suppressing extra-family fit.",
        },
        "interpretation_notes": (
            "HistGradientBoostingRegressor with simple ratio features and a 24-cluster KMeans geo "
            "bucket on lat/lon. Best of a 3-config 5-fold sweep on learning_rate/max_iter/max_leaf_nodes. "
            "Predictions clipped to the observed [0.15, 5.0] y-range that the synthetic generator inherits "
            "from the original California Housing dataset."
        ),
        "wall_time_seconds": round(time.time() - t0, 2),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{time.time() - t0:5.1f}s] wrote result.json (RMSE={best_rmse:.5f})")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
    os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
    main()
