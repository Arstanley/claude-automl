"""Solution for playground-series-s3e8 (gemstone price prediction, RMSE).

Skills applied:
  - skill_tune_when_n_train_supports_it: n_train=154858 >> 500 and no
    size/latency/production tokens -> run a small 4-config grid.
  - skill_seed_bag_final_regressor: regression target with nunique>>20,
    n>=500, HGB takes random_state, no latency tokens -> 3-seed bag the
    final refit.

Other skill checks:
  - missing_indicator: no NaN in train or test -> N/A.
  - stringify_low_cardinality_int_codes: no int-coded categoricals -> N/A.
  - probe_group_structure_in_ids: ids are dense disjoint integers, no
    parseable structure -> N/A.
  - verify_split_claim_before_holdout: prompt makes no structural split
    claim -> N/A.
  - train_on_listed_auxiliary_files: prompt mentions an original gemstone
    dataset but it is not provided in DATA_DIR -> N/A.
  - multi_axis_constraint_budgets: single-axis (RMSE), no latency/size
    budgets -> N/A.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e8/data")

CAT_COLS = ["cut", "color", "clarity"]
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def featurize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Ordinal-encode the three categoricals based on train union test, then
    # let HGB treat them as categorical features.
    out_tr = train.copy()
    out_te = test.copy()
    for c in CAT_COLS:
        cats = pd.Index(sorted(set(train[c].unique()) | set(test[c].unique())))
        m = {v: i for i, v in enumerate(cats)}
        out_tr[c] = out_tr[c].map(m).astype("int32")
        out_te[c] = out_te[c].map(m).astype("int32")
    # x,y,z == 0 is physically meaningless: treat as NaN so HGB handles them.
    for c in ("x", "y", "z"):
        out_tr.loc[out_tr[c] == 0, c] = np.nan
        out_te.loc[out_te[c] == 0, c] = np.nan
    feat_cols = CAT_COLS + NUM_COLS
    return out_tr[feat_cols], out_te[feat_cols]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_model(params: dict, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=seed,
        categorical_features=[True, True, True, False, False, False, False, False, False],
        **params,
    )


def cv_score(X: pd.DataFrame, y: np.ndarray, params: dict, n_splits: int = 4) -> float:
    """OOF RMSE in original price units, training on log(price)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros(len(X))
    log_y = np.log1p(y)
    for tr_idx, va_idx in kf.split(X):
        m = make_model(params, seed=0)
        m.fit(X.iloc[tr_idx], log_y[tr_idx])
        oof[va_idx] = np.expm1(m.predict(X.iloc[va_idx]))
    oof = np.clip(oof, a_min=1.0, a_max=None)
    return rmse(y, oof)


def main() -> None:
    t0 = time.time()
    train_raw, test_raw = load()
    X_full, X_test = featurize(train_raw, test_raw)
    y_full = train_raw["price"].to_numpy()
    test_ids = test_raw["id"].to_numpy()

    # skill_tune_when_n_train_supports_it: 4-config grid
    grid = [
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 63, "min_samples_leaf": 40, "l2_regularization": 0.0},
        {"learning_rate": 0.03, "max_iter": 1000, "max_leaf_nodes": 63, "min_samples_leaf": 30, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 127, "min_samples_leaf": 20, "l2_regularization": 0.1},
    ]
    cv_results = []
    for p in grid:
        s = cv_score(X_full, y_full, p, n_splits=4)
        cv_results.append({"params": p, "cv_rmse": s})
        print(f"  cfg {p} -> oof RMSE {s:.3f}  (t={time.time()-t0:.1f}s)")

    best = min(cv_results, key=lambda r: r["cv_rmse"])
    best_params = best["params"]
    best_cv = best["cv_rmse"]
    print(f"BEST cfg cv_rmse={best_cv:.3f} params={best_params}")

    # skill_seed_bag_final_regressor: refit on full train with 3 seeds
    log_y_full = np.log1p(y_full)
    seeds = [0, 1, 2]
    test_log_preds = []
    for s in seeds:
        m = make_model(best_params, seed=s)
        m.fit(X_full, log_y_full)
        test_log_preds.append(m.predict(X_test))
    test_pred = np.expm1(np.mean(test_log_preds, axis=0))
    test_pred = np.clip(test_pred, a_min=1.0, a_max=None)

    # Write submission
    sub = pd.DataFrame({"id": test_ids, "price": test_pred})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": "price",
        "headline_metric": "RMSE",
        "headline_value": best_cv,
        "all_metrics": {
            "cv_rmse_best": best_cv,
            "cv_rmse_all_configs": [{"params": r["params"], "cv_rmse": r["cv_rmse"]} for r in cv_results],
            "n_train": int(len(train_raw)),
            "n_test": int(len(test_raw)),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_seed_bag_final_regressor.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor on log1p(price) with cut/color/clarity "
            "as native categoricals; 4-config grid tuned by 4-fold OOF RMSE; final "
            "fit seed-bagged over [0,1,2]. x/y/z==0 dimensions treated as missing."
        ),
        "audit": {
            "seed_bag": seeds,
            "best_params": best_params,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done in {time.time()-t0:.1f}s. submission rows={len(sub)} best_cv={best_cv:.3f}")


if __name__ == "__main__":
    main()
