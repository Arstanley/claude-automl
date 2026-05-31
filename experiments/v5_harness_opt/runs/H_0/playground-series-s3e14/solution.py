"""
Playground Series S3E14 — Wild Blueberry Yield Prediction.

Tabular regression, n_train=12231, 16 features, metric=MAE.

Skills applied:
- skill_tune_when_n_train_supports_it: GBM family, n_train>>500, no
  size/latency constraints in prompt -> a small explicit grid is appropriate.
- skill_verify_split_claim_before_holdout: prompt makes no temporal/group
  claim about the split, so plain KFold is appropriate.

Model: sklearn HistGradientBoostingRegressor with MAE loss
(lightgbm/xgboost not installed). Tight budget: single-thread per model,
3-config grid via a single 80/20 holdout, final refit on all data.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# Limit thread fan-out: the host has many sibling jobs competing for cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e14/data")
HERE = Path(__file__).resolve().parent

RNG = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t0 = time.time()
    log(f"[load] reading data ...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log(f"[load] train {train.shape} test {test.shape} dt={time.time()-t0:.1f}s")

    target = "yield"
    id_col = "id"
    feats = [c for c in train.columns if c not in (target, id_col)]

    X = train[feats].values.astype(np.float32)
    y = train[target].values.astype(np.float64)
    X_test = test[feats].values.astype(np.float32)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RNG,
    )

    grid = [
        dict(max_iter=400, learning_rate=0.05, max_depth=6,
             min_samples_leaf=20, l2_regularization=0.0,
             loss="absolute_error"),
        dict(max_iter=600, learning_rate=0.03, max_depth=7,
             min_samples_leaf=30, l2_regularization=0.1,
             loss="absolute_error"),
        dict(max_iter=400, learning_rate=0.05, max_depth=6,
             min_samples_leaf=20, l2_regularization=0.0,
             loss="squared_error"),
    ]

    grid_results = []
    for i, params in enumerate(grid):
        t1 = time.time()
        m = HistGradientBoostingRegressor(
            random_state=RNG, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            **params,
        )
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        mae = float(np.mean(np.abs(pred - y_va)))
        grid_results.append({"params": params, "holdout_mae": mae,
                             "fit_sec": round(time.time() - t1, 1)})
        log(f"[grid {i}] loss={params['loss']:>15s} d={params['max_depth']} "
            f"it={params['max_iter']} -> MAE {mae:.4f} ({time.time()-t1:.1f}s)")

    grid_results.sort(key=lambda r: r["holdout_mae"])
    best = grid_results[0]
    log(f"[grid] best holdout MAE {best['holdout_mae']:.4f}")

    # Refit best on full train, average across 3 seeds for stability.
    log(f"[refit] full-train bag of 3 seeds ...")
    test_preds = np.zeros(len(test), dtype=np.float64)
    n_seeds = 3
    for seed in range(n_seeds):
        t1 = time.time()
        m = HistGradientBoostingRegressor(
            random_state=seed, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            **best["params"],
        )
        m.fit(X, y)
        test_preds += m.predict(X_test) / n_seeds
        log(f"[refit] seed {seed} done ({time.time()-t1:.1f}s)")

    sub = pd.DataFrame({id_col: test[id_col].values, target: test_preds})
    sub.to_csv(HERE / "submission.csv", index=False)
    log(f"[write] submission.csv ({len(sub)} rows)")

    result = {
        "target_column_chosen": target,
        "headline_metric": "MAE",
        "headline_value": best["holdout_mae"],
        "all_metrics": {
            "holdout_mae_best_cfg": best["holdout_mae"],
            "grid": [
                {"params": g["params"], "holdout_mae": g["holdout_mae"]}
                for g in grid_results
            ],
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_verify_split_claim_before_holdout.md",
        ],
        "interpretation_notes": (
            "Tabular regression on Wild Blueberry yield (n_train=12231, no "
            "NaNs, 16 numeric features). Compared 3 HistGradientBoosting "
            "configs on a single 80/20 holdout (MAE-loss vs squared-loss). "
            "Final = 3-seed bag of best config refit on full train. "
            "Used single 80/20 holdout (not 5-fold CV) to stay under the "
            "3-minute budget given host contention."
        ),
        "audit": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": len(feats),
            "grid": grid_results,
            "chosen_params": best["params"],
            "n_seeds": n_seeds,
            "runtime_sec": round(time.time() - t0, 1),
        },
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2))
    log(f"[done] wrote result.json, total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
