"""
playground-series-s3e14: Wild Blueberry Yield Prediction (regression, MAE).

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=12231 >= 500, no latency/size tokens
  in prompt -> small explicit grid over HistGradientBoostingRegressor.
- Family-comparison rule (linear Ridge vs GBM): n_train >= 500, no constraints.
  Submit simple average if mean CV scores differ by < 0.5% relative; else winner.

n_jobs=2, 2 minute budget, small grid.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e14/data")
HERE = Path(__file__).parent.resolve()
N_JOBS = 2
SEED = 42
T0 = time.time()


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target = "yield"
    id_col = "id"
    feat_cols = [c for c in train.columns if c not in (target, id_col)]

    X = train[feat_cols].copy()
    y = train[target].values
    X_test = test[feat_cols].copy()

    # All features are numeric floats; no NaNs (verified). Simple pipeline.
    numeric_pipe_linear = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    pre_linear = ColumnTransformer([("num", numeric_pipe_linear, feat_cols)])
    pre_gbm = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feat_cols)]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scorer = "neg_mean_absolute_error"

    # ---- Linear family: Ridge with small alpha grid ----
    ridge_grid = [0.1, 1.0, 10.0, 100.0]
    ridge_results = []
    for alpha in ridge_grid:
        pipe = Pipeline(
            [("pre", pre_linear), ("est", Ridge(alpha=alpha, random_state=SEED))]
        )
        scores = cross_val_score(pipe, X, y, scoring=scorer, cv=cv, n_jobs=N_JOBS)
        ridge_results.append((alpha, -scores.mean(), -scores.std()))
        print(f"[ridge alpha={alpha}] MAE={-scores.mean():.4f} +/- {-scores.std():.4f}")

    ridge_results.sort(key=lambda r: r[1])
    best_ridge_alpha, ridge_mae, _ = ridge_results[0]

    # ---- GBM family: HistGradientBoostingRegressor with small grid ----
    # Small grid: vary learning_rate x max_leaf_nodes; fix loss=absolute_error for MAE.
    gbm_grid = [
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 400},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400},
        {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300},
        {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 300},
    ]
    gbm_results = []
    for params in gbm_grid:
        est = HistGradientBoostingRegressor(
            loss="absolute_error",
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            **params,
        )
        pipe = Pipeline([("pre", pre_gbm), ("est", est)])
        scores = cross_val_score(pipe, X, y, scoring=scorer, cv=cv, n_jobs=N_JOBS)
        gbm_results.append((params, -scores.mean(), -scores.std()))
        print(f"[gbm {params}] MAE={-scores.mean():.4f} +/- {-scores.std():.4f}")
        if time.time() - T0 > 100:
            print("[budget] stopping gbm grid early")
            break

    gbm_results.sort(key=lambda r: r[1])
    best_gbm_params, gbm_mae, _ = gbm_results[0]

    # ---- Family comparison ----
    rel_diff = abs(ridge_mae - gbm_mae) / max(abs(ridge_mae), abs(gbm_mae))
    print(f"ridge_MAE={ridge_mae:.4f}  gbm_MAE={gbm_mae:.4f}  rel_diff={rel_diff:.4f}")

    # Refit best models on full train, predict on test.
    ridge_best = Pipeline(
        [("pre", pre_linear), ("est", Ridge(alpha=best_ridge_alpha, random_state=SEED))]
    )
    ridge_best.fit(X, y)
    pred_ridge = ridge_best.predict(X_test)

    gbm_best = Pipeline(
        [
            ("pre", pre_gbm),
            (
                "est",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    **best_gbm_params,
                ),
            ),
        ]
    )
    gbm_best.fit(X, y)
    pred_gbm = gbm_best.predict(X_test)

    if rel_diff < 0.005:
        final_pred = 0.5 * (pred_ridge + pred_gbm)
        chosen = "average(ridge,gbm)"
        headline = 0.5 * (ridge_mae + gbm_mae)
    elif ridge_mae < gbm_mae:
        final_pred = pred_ridge
        chosen = f"ridge(alpha={best_ridge_alpha})"
        headline = ridge_mae
    else:
        final_pred = pred_gbm
        chosen = f"hgbr({best_gbm_params})"
        headline = gbm_mae

    # ---- Submission ----
    sub = pd.DataFrame({id_col: test[id_col].values, target: final_pred})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": target,
        "headline_metric": "MAE",
        "headline_value": float(headline),
        "all_metrics": {
            "ridge_cv_mae": float(ridge_mae),
            "gbm_cv_mae": float(gbm_mae),
            "ridge_best_alpha": float(best_ridge_alpha),
            "gbm_best_params": best_gbm_params,
            "family_rel_diff": float(rel_diff),
            "chosen": chosen,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Regression on 12231 rows, 16 all-numeric features, no NaNs. Compared Ridge "
            "(linear, scaled) vs HistGradientBoostingRegressor (loss=absolute_error to "
            "match MAE) under shared 5-fold CV with small explicit grids. Applied family-"
            "comparison rule: average if relative gap <0.5%, else pick winner."
        ),
        "audit": {
            "family_comparison": {
                "ridge": {"mean_cv_mae": float(ridge_mae), "alpha": float(best_ridge_alpha)},
                "gbm": {"mean_cv_mae": float(gbm_mae), "params": best_gbm_params},
                "rel_diff": float(rel_diff),
                "threshold": 0.005,
                "decision": chosen,
            }
        },
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2))
    print(f"wrote submission.csv ({len(sub)} rows) and result.json; headline MAE={headline:.4f}")
    print(f"elapsed={time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
