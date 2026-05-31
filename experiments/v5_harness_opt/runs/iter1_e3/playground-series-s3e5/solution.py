"""
playground-series-s3e5 — Wine Quality (ordinal target, QWK metric).

Strategy:
  - n=1644, 11 numeric features, no missing data, no constraint tokens => apply
    skill_tune_when_n_train_supports_it (small explicit grid) under family-comparison rule.
  - Regression-then-round on the ordinal target is the standard, higher-QWK approach.
  - Compared families under shared 5-fold CV: Ridge baseline vs HistGradientBoostingRegressor (4 configs).
  - Blend if relative diff < 0.005, else pick winner.

Robust to CPU contention: single-threaded (no joblib parallelism, no early-stopping
validation split), small max_iter — task is tiny so this is plenty.
"""
from __future__ import annotations

# Pin threads BEFORE any sklearn import — survives oversubscribed sandboxes.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e5/data")
WORK_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/iter1_e3/playground-series-s3e5")

SEED = 42
N_SPLITS = 5
TARGET = "quality"
ID_COL = "Id"
TARGET_LO, TARGET_HI = 3, 8


def qwk(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.clip(np.rint(np.asarray(y_pred)).astype(int), TARGET_LO, TARGET_HI)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def cv_eval(estimator_factory, X, y, n_splits=N_SPLITS, seed=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for tr_idx, va_idx in kf.split(X):
        est = estimator_factory()
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = est.predict(X.iloc[va_idx])
        oof[va_idx] = p
        fold_scores.append(qwk(y.iloc[va_idx], p))
    return float(np.mean(fold_scores)), oof, fold_scores


def fit_full_and_predict(estimator_factory, X, y, X_test):
    est = estimator_factory()
    est.fit(X, y)
    return est.predict(X_test)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X = train[feats].copy()
    y = train[TARGET].astype(int).copy()
    X_test = test[feats].copy()

    # --- Ridge baseline ---
    def make_ridge():
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=SEED))])

    ridge_cv, ridge_oof, ridge_fold_scores = cv_eval(make_ridge, X, y)
    print(f"[{time.time()-t0:.1f}s] ridge_cv_qwk={ridge_cv:.4f}", flush=True)

    # --- HGB small explicit grid (no early stopping / no internal threads) ---
    hgb_grid = [
        dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0),
        dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0),
        dict(learning_rate=0.08, max_iter=200, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.0),
    ]
    hgb_results = []
    for i, cfg in enumerate(hgb_grid):
        def factory(cfg=cfg):
            return HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **cfg)
        score, oof, folds = cv_eval(factory, X, y)
        hgb_results.append({"cfg": cfg, "cv": score, "fold_scores": folds, "oof": oof})
        print(f"[{time.time()-t0:.1f}s] hgb[{i}] cv_qwk={score:.4f} cfg={cfg}", flush=True)

    hgb_results.sort(key=lambda r: r["cv"], reverse=True)
    best_hgb = hgb_results[0]
    hgb_cv = best_hgb["cv"]

    # --- Family comparison rule ---
    rel_diff = abs(ridge_cv - hgb_cv) / max(abs(ridge_cv), abs(hgb_cv))
    blend = rel_diff < 0.005

    def make_best_hgb():
        return HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **best_hgb["cfg"])

    hgb_test = fit_full_and_predict(make_best_hgb, X, y, X_test)
    ridge_test = fit_full_and_predict(make_ridge, X, y, X_test)

    if blend:
        final_cont = 0.5 * (hgb_test + ridge_test)
        oof_blend = 0.5 * (ridge_oof + best_hgb["oof"])
        headline = qwk(y, oof_blend)
        chosen = "blend(ridge,hgb)"
    elif hgb_cv >= ridge_cv:
        final_cont = hgb_test
        headline = hgb_cv
        chosen = "hgb"
    else:
        final_cont = ridge_test
        headline = ridge_cv
        chosen = "ridge"

    final_pred = np.clip(np.rint(final_cont).astype(int), TARGET_LO, TARGET_HI)

    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: final_pred})
    sub.to_csv(WORK_DIR / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "quadratic_weighted_kappa",
        "headline_value": float(headline),
        "all_metrics": {
            "ridge_cv_qwk": float(ridge_cv),
            "best_hgb_cv_qwk": float(hgb_cv),
            "chosen_model": chosen,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Regression-then-round on ordinal quality (3..8) yields higher QWK than multiclass on n=1644. "
            "Family-comparison rule under shared 5-fold CV: Ridge baseline vs 4-config HistGradientBoostingRegressor "
            f"(skill_tune_when_n_train_supports_it). Chose '{chosen}' (rel_diff={rel_diff:.4f}, threshold 0.005)."
        ),
        "audit": {
            "family_comparison": {
                "ridge_cv_qwk": float(ridge_cv),
                "hgb_cv_qwk": float(hgb_cv),
                "rel_diff": float(rel_diff),
                "blend_threshold": 0.005,
                "blended": bool(blend),
                "chosen": chosen,
                "hgb_grid": [
                    {"cfg": r["cfg"], "cv": float(r["cv"]), "fold_scores": [float(s) for s in r["fold_scores"]]}
                    for r in hgb_results
                ],
                "ridge_fold_scores": [float(s) for s in ridge_fold_scores],
            },
            "runtime_sec": round(time.time() - t0, 2),
        },
    }
    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{time.time()-t0:.1f}s] DONE chosen={chosen} headline={headline:.4f}", flush=True)


if __name__ == "__main__":
    main()
