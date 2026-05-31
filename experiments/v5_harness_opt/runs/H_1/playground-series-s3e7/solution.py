"""Solution for playground-series-s3e7: predict booking_status (binary, AUC).

Skills applied:
 - skill_tune_when_n_train_supports_it: small explicit grid over HistGradientBoosting (n_train=33680, no constraint tokens beyond training budget).
 - skill_blend_two_model_families: HistGradientBoosting + ExtraTrees, blend weight chosen on OOF.

Skills evaluated but not applied: see result.json.skills_evaluated_not_applied.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

WORKING_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e7/data")
N_JOBS = 2
SEED = 42
N_SPLITS = 5
TARGET = "booking_status"
ID_COL = "id"


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple derived features that often help on this style of booking data.
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    # Avoid div-by-zero
    df["price_per_guest"] = df["avg_price_per_room"] / df["total_guests"].clip(lower=1)
    df["price_per_night"] = df["avg_price_per_room"] * df["total_nights"]
    df["had_previous_cancel"] = (df["no_of_previous_cancellations"] > 0).astype(int)
    df["had_previous_booking"] = (df["no_of_previous_bookings_not_canceled"] > 0).astype(int)
    # Day-of-year-ish proxy
    df["arrival_doy"] = (df["arrival_month"] - 1) * 31 + df["arrival_date"]
    return df


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    y = train[TARGET].astype(int).values
    test_ids = test[ID_COL].values

    train_fe = feature_engineer(train.drop(columns=[TARGET]))
    test_fe = feature_engineer(test)

    feature_cols = [c for c in train_fe.columns if c != ID_COL]
    X = train_fe[feature_cols].values
    X_test = test_fe[feature_cols].values

    # ----- Skill: tune_when_n_train_supports_it -----
    # Small explicit grid for HistGradientBoosting, CV-selected.
    hgb_grid = [
        {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"learning_rate": 0.03, "max_iter": 800, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 1.0},
        {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    ]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def cv_hgb(params):
        oof = np.zeros(len(X))
        for tr_idx, va_idx in skf.split(X, y):
            m = HistGradientBoostingClassifier(
                random_state=SEED,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                **params,
            )
            m.fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        return roc_auc_score(y, oof), oof

    best_score = -np.inf
    best_params = None
    best_oof_hgb = None
    grid_scores = []
    for params in hgb_grid:
        s, oof = cv_hgb(params)
        grid_scores.append({"params": params, "auc": float(s)})
        if s > best_score:
            best_score = s
            best_params = params
            best_oof_hgb = oof
        print(f"HGB {params} -> AUC {s:.5f}")

    # Fit final HGB on full train
    final_hgb = HistGradientBoostingClassifier(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        **best_params,
    )
    final_hgb.fit(X, y)
    pred_hgb_test = final_hgb.predict_proba(X_test)[:, 1]

    # ----- Skill: blend_two_model_families -----
    # ExtraTrees as the second family. CV OOF to pick blend weight.
    oof_et = np.zeros(len(X))
    et_test = np.zeros(len(X_test))
    for tr_idx, va_idx in skf.split(X, y):
        m = ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=N_JOBS,
            random_state=SEED,
        )
        m.fit(X[tr_idx], y[tr_idx])
        oof_et[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        et_test += m.predict_proba(X_test)[:, 1] / N_SPLITS

    auc_et = roc_auc_score(y, oof_et)
    auc_hgb = roc_auc_score(y, best_oof_hgb)
    print(f"OOF AUC HGB={auc_hgb:.5f}  ET={auc_et:.5f}")

    # Sweep blend weight w on HGB, (1-w) on ET
    best_w, best_blend = 1.0, auc_hgb
    blend_scores = []
    for w in np.linspace(0.0, 1.0, 11):
        blend = w * best_oof_hgb + (1.0 - w) * oof_et
        s = roc_auc_score(y, blend)
        blend_scores.append({"w_hgb": float(w), "auc": float(s)})
        if s > best_blend:
            best_blend = s
            best_w = float(w)
    print(f"Best blend w_hgb={best_w:.2f} -> AUC {best_blend:.5f}")

    if best_w in (0.0, 1.0):
        print("Blend collapses to single model; shipping single model.")

    final_test = best_w * pred_hgb_test + (1.0 - best_w) * et_test

    # Write submission
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: final_test})
    sub_path = WORKING_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    elapsed = time.time() - t0

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "roc_auc",
        "headline_value": float(best_blend),
        "all_metrics": {
            "oof_auc_hgb": float(auc_hgb),
            "oof_auc_extratrees": float(auc_et),
            "oof_auc_blend": float(best_blend),
            "blend_weight_hgb": best_w,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_blend_two_model_families.md",
        ],
        "skills_evaluated_not_applied": [
            "skill_probe_group_structure_in_ids.md: no train/test id overlap (0 ids shared) and no compound id format",
            "skill_train_on_listed_auxiliary_files.md: prompt mentions original Reservation Cancellation Prediction dataset but no aux file present in DATA_DIR",
            "skill_verify_split_claim_before_holdout.md: prompt makes no structural (temporal/group) split claim",
            "skill_stringify_low_cardinality_int_codes.md: pipeline is tree-based (HGB+ExtraTrees); no scaler/linear/distance model in use",
            "skill_multi_axis_constraint_budgets.md: only a single training-time constraint axis, no unquantified axes",
            "skill_missing_indicator_when_missingness_signals_class.md: zero NaNs in train and test",
            "skill_tune_decision_threshold_for_f1.md: metric is ROC AUC (rank-based), not F1/F-beta/MCC; threshold tuning irrelevant",
            "skill_log_target_for_skewed_nonnegative_regression.md: task is binary classification, not regression",
        ],
        "best_hgb_params": best_params,
        "grid_scores": grid_scores,
        "blend_scores": blend_scores,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "elapsed_sec": elapsed,
    }
    with open(WORKING_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {sub_path} and result.json. Elapsed {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
