"""
Solution for Kaggle Playground Series S3E9 — Concrete Strength regression.

Task: predict Strength from 8 numeric features. Metric: RMSE.
Train n=4325, test n=1082, no missing values, all numeric features.

Skill library application (v5_harness_opt):
  - skill_tune_when_n_train_supports_it.md  -> APPLIES (n=4325 >= 500, GBM family,
        no size/latency tokens in prompt). Run small explicit grid.
  - skill_stringify_low_cardinality_int_codes.md -> OPTIONAL only (pipeline is
        tree-based HistGradientBoosting; AgeInDays carries magnitude semantics
        (1..365 days of curing), so we keep it numeric.
  - skill_probe_group_structure_in_ids.md   -> does NOT apply (id has 0
        train/test overlap; no compound IDs; no other group-like cols).
  - skill_missing_indicator_when_missingness_signals_class.md -> does NOT apply
        (no missing values).
  - other skills: precondition fails (no aux files, no split claim, no
        multi-axis constraint prompt).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e9/data")

RANDOM_STATE = 0
N_SPLITS = 5


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target = "Strength"
    id_col = "id"
    feature_cols = [c for c in train.columns if c not in (target, id_col)]

    X = train[feature_cols].values
    y = train[target].values
    X_test = test[feature_cols].values
    test_ids = test[id_col].values

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Skill 7: small explicit grid (3 configs) -----------------------------
    grid = [
        dict(max_iter=500, learning_rate=0.05, max_depth=None,
             max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=800, learning_rate=0.03, max_depth=None,
             max_leaf_nodes=63, min_samples_leaf=15, l2_regularization=0.1),
        dict(max_iter=1000, learning_rate=0.02, max_depth=None,
             max_leaf_nodes=31, min_samples_leaf=10, l2_regularization=0.0),
    ]

    grid_results = []
    for params in grid:
        model = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, early_stopping=False, **params,
        )
        # use neg_root_mean_squared_error -> negate
        scores = -cross_val_score(
            model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        grid_results.append((float(scores.mean()), float(scores.std()), params))
        print(f"  HGB {params}: CV RMSE = {scores.mean():.4f} ± {scores.std():.4f}")

    grid_results.sort(key=lambda t: t[0])  # ascending RMSE (lower is better)
    best_rmse_cv, best_std, best_params = grid_results[0]
    print(f"\nBest HGB config: {best_params}  (CV RMSE {best_rmse_cv:.4f})")

    # ---- Fit best HGB on full train, then a small bagging ensemble across seeds
    seeds = [0, 1, 2, 3, 4]
    test_preds_per_seed = []
    oof = np.zeros(len(X))

    # Out-of-fold for the chosen config (seed 0) for an honest holdout estimate
    fold_oof_rmses = []
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X)):
        model = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, early_stopping=False, **best_params,
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X[va_idx])
        fold_oof_rmses.append(rmse(y[va_idx], oof[va_idx]))
    oof_rmse = rmse(y, oof)
    print(f"OOF RMSE (best config): {oof_rmse:.4f}  per-fold: {[round(r,4) for r in fold_oof_rmses]}")

    # Ensemble across random seeds on full data
    for s in seeds:
        model = HistGradientBoostingRegressor(
            random_state=s, early_stopping=False, **best_params,
        )
        model.fit(X, y)
        test_preds_per_seed.append(model.predict(X_test))
    test_pred = np.mean(test_preds_per_seed, axis=0)

    # Clip to observed strength range (no negative concrete strength)
    test_pred = np.clip(test_pred, 0.0, None)

    # ---- write submission -----------------------------------------------------
    sub = pd.DataFrame({id_col: test_ids, target: test_pred})
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"wrote {HERE / 'submission.csv'}  rows={len(sub)}")

    # ---- write result.json ----------------------------------------------------
    result = {
        "target_column_chosen": target,
        "headline_metric": "RMSE",
        "headline_value": oof_rmse,
        "all_metrics": {
            "cv_rmse_mean": best_rmse_cv,
            "cv_rmse_std": best_std,
            "oof_rmse": oof_rmse,
            "per_fold_rmse": fold_oof_rmses,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "skills_evaluated_not_applied": {
            "skill_probe_group_structure_in_ids.md":
                "no train/test id overlap, no compound IDs, no group cols",
            "skill_stringify_low_cardinality_int_codes.md":
                "AgeInDays has 15 unique ints but encodes magnitude (1..365 day "
                "curing time); HGB tree pipeline handles it correctly as numeric",
            "skill_missing_indicator_when_missingness_signals_class.md":
                "no missing values in train or test",
            "skill_train_on_listed_auxiliary_files.md":
                "no auxiliary files in DATA_DIR",
            "skill_verify_split_claim_before_holdout.md":
                "prompt makes no split structure claim",
            "skill_multi_axis_constraint_budgets.md":
                "no latency/size/production constraints in prompt",
        },
        "hyperparameter_grid": [
            {"params": p, "cv_rmse_mean": m, "cv_rmse_std": s}
            for (m, s, p) in grid_results
        ],
        "chosen_config": best_params,
        "seed_ensemble_seeds": seeds,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_cols": feature_cols,
        "interpretation_notes": (
            "Tabular regression, n=4325. Picked HistGradientBoostingRegressor; "
            "tuned over a 3-config explicit grid via 5-fold KFold CV (skill: "
            "tune_when_n_train_supports_it). Best config refit on full train "
            "with 5-seed bagging; predictions clipped to >= 0."
        ),
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {HERE / 'result.json'}")
    print(f"\nheadline_metric: RMSE = {oof_rmse:.4f}")


if __name__ == "__main__":
    main()
