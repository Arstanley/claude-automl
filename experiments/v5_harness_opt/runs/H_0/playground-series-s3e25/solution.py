"""
playground-series-s3e25 — Mohs Hardness regression.
Metric: Median Absolute Error (MedAE). Target: Hardness (continuous, ~1..10).

Skills applied:
  - skill_tune_when_n_train_supports_it.md : n_train=8325 >> 500, no constraint
    tokens (size/latency/MB/ms/production/mobile/p99) in prompt, GBM model
    family. Use small explicit 3-config grid with KFold CV.

Skills checked but NOT applied:
  - skill_missing_indicator_when_missingness_signals_class : no NaN in data.
  - skill_probe_group_structure_in_ids : heavy value overlap on feature
    columns (94-100%) but these are derived chemical descriptors that
    legitimately repeat across compounds (id is unique). Not a CV leak.
  - skill_stringify_low_cardinality_int_codes : all features are float.
  - skill_verify_split_claim_before_holdout : prompt makes no split claim.
  - skill_multi_axis_constraint_budgets : no constraints stated.
  - skill_train_on_listed_auxiliary_files : prompt mentions an original dataset
    but does not provide it in DATA_DIR; cannot use.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e25/data")
ID_COL = "id"
TARGET = "Hardness"


def medae(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train={train.shape}  test={test.shape}")

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X = train[feature_cols].values
    y = train[TARGET].values
    X_test = test[feature_cols].values

    # --- skill_tune_when_n_train_supports_it: small explicit grid (3 configs) ---
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
             min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.03, max_depth=None, max_leaf_nodes=63,
             min_samples_leaf=20, l2_regularization=0.1),
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
             min_samples_leaf=40, l2_regularization=1.0),
    ]
    # Note: MedAE is robust to outliers and aligned with quantile/absolute losses.
    # HGB supports loss='absolute_error' which optimises a related criterion;
    # we use it because the metric is an absolute-error quantile.
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    results = []
    for params in grid:
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X)):
            m = HistGradientBoostingRegressor(
                loss="absolute_error", random_state=0, **params,
            )
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            fold_scores.append(medae(y[va_idx], pred))
        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        results.append((mean_s, std_s, params, fold_scores))
        print(f"  cfg {params}: MedAE={mean_s:.4f} ± {std_s:.4f}")

    # Lower MedAE is better.
    results.sort(key=lambda r: r[0])
    best_mean, best_std, best_params, best_folds = results[0]
    print(f"chosen: {best_params}  CV MedAE={best_mean:.4f} ± {best_std:.4f}")

    # Refit on full training data.
    final = HistGradientBoostingRegressor(
        loss="absolute_error", random_state=0, **best_params,
    )
    final.fit(X, y)
    test_pred = final.predict(X_test)
    # Clip to observed target range (target is bounded by Mohs scale 1..10).
    test_pred = np.clip(test_pred, float(y.min()), float(y.max()))

    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  ({len(sub)} rows)")

    # Sanity check on holdout-style metric (use mean CV score as headline).
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "MedAE",
        "headline_value": best_mean,
        "all_metrics": {
            "cv_medae_mean": best_mean,
            "cv_medae_std": best_std,
            "cv_medae_folds": best_folds,
            "grid": [
                {
                    "params": r[2],
                    "cv_medae_mean": r[0],
                    "cv_medae_std": r[1],
                }
                for r in results
            ],
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor with loss='absolute_error' (aligned with "
            "MedAE metric). 3-config explicit grid + 5-fold KFold CV selected the "
            "config with lowest mean MedAE; refit on full train. Features are 11 "
            "numeric chemical descriptors; no missingness, no group leak (id is "
            "unique). Predictions clipped to observed target range."
        ),
        "audit": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "chosen_config": best_params,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    }
    res_path = HERE / "result.json"
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {res_path}  headline MedAE={best_mean:.4f}  elapsed={result['audit']['elapsed_sec']}s")


if __name__ == "__main__":
    main()
