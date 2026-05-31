"""
Playground Series S3E11 — Media Campaign Cost (regression, RMSLE).

Skills applied (from harness/skills):
- skill_tune_when_n_train_supports_it: n_train=288K, GBM family, no size/latency
  constraints in the prompt -> small explicit grid with held-out validation.
- skill_probe_group_structure_in_ids: cheap check on `id` overlap between train/test
  (synthetic data so expected empty; verified, not pursued further).

Approach: HistGradientBoostingRegressor on log1p(cost) so that minimizing
in-model squared loss aligns with the RMSLE evaluation metric. Single
80/20 holdout (fast on 288K rows), early stopping inside the model,
tiny 2-config grid over learning_rate (0.05 vs 0.1) with max_iter=600 and
n_iter_no_change=25. Best config refit on full train.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/"
    "playground-series-s3e11/data"
)

TARGET = "cost"
ID = "id"


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # sklearn.mean_squared_log_error(..., squared=False) — implemented locally to
    # avoid version-specific kwarg differences.
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    diff = np.log1p(y_true) - np.log1p(y_pred)
    return float(np.sqrt(np.mean(diff * diff)))


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # skill_probe_group_structure_in_ids: any overlap between train/test ids?
    overlap = len(set(train[ID]).intersection(set(test[ID])))
    print(f"train/test id overlap: {overlap}")

    feature_cols = [c for c in train.columns if c not in (ID, TARGET)]
    X = train[feature_cols].to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy(dtype=np.float64)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    y_log = np.log1p(y)

    X_tr, X_va, y_tr, y_va, y_tr_log, y_va_log = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42
    )

    # Tiny grid — 2 configs only, per budget.
    grid = [
        {"learning_rate": 0.10, "max_leaf_nodes": 63},
        {"learning_rate": 0.05, "max_leaf_nodes": 127},
    ]

    base_kw = dict(
        loss="squared_error",
        max_iter=600,
        min_samples_leaf=40,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=0,
    )

    results = []
    for i, cfg in enumerate(grid):
        m = HistGradientBoostingRegressor(**base_kw, **cfg)
        m.fit(X_tr, y_tr_log)
        pred_va = np.expm1(m.predict(X_va))
        score = rmsle(y_va, pred_va)
        n_iter = int(m.n_iter_)
        print(f"cfg {i}: {cfg} -> RMSLE={score:.5f}  n_iter={n_iter}")
        results.append({"cfg": cfg, "rmsle": score, "n_iter": n_iter})

    best = min(results, key=lambda r: r["rmsle"])
    print(f"best: {best}")

    # Refit on full train with the chosen config; reuse learned n_iter
    # (no early stopping on the final fit, since we don't carve out another val).
    final_iter = max(50, best["n_iter"])  # safety floor
    final_kw = dict(base_kw)
    final_kw["early_stopping"] = False
    final_kw.pop("validation_fraction", None)
    final_kw.pop("n_iter_no_change", None)
    final_kw["max_iter"] = final_iter
    final = HistGradientBoostingRegressor(**final_kw, **best["cfg"])
    final.fit(X, y_log)
    pred_test = np.clip(np.expm1(final.predict(X_test)), a_min=0.0, a_max=None)

    submission = pd.DataFrame({ID: test[ID].values, TARGET: pred_test})
    submission.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "RMSLE",
        "headline_value": best["rmsle"],
        "all_metrics": {
            "holdout_RMSLE_best": best["rmsle"],
            "grid": [
                {"cfg": r["cfg"], "rmsle": r["rmsle"], "n_iter": r["n_iter"]}
                for r in results
            ],
            "final_n_iter": final_iter,
            "id_overlap_train_test": overlap,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "wall_seconds": round(time.time() - t0, 1),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_probe_group_structure_in_ids.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor on log1p(cost) to align squared-error "
            "training with RMSLE evaluation. Tiny 2-config grid over "
            "(learning_rate, max_leaf_nodes) with early stopping on a 15% "
            "internal validation slice; chosen config refit on full train using "
            "the learned iteration count."
        ),
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2))
    print(f"wrote submission.csv ({len(submission)} rows) and result.json")
    print(f"wall: {result['all_metrics']['wall_seconds']}s")


if __name__ == "__main__":
    main()
