"""
Solution for playground-series-s4e3: Steel Plates Faults multi-label classification.

Task: predict probability of 7 binary defect categories. Metric: mean column-wise ROC AUC.

Approach (budget: 3 min, n_jobs=2, small grid, n_train=15K):
- Multi-label: train one binary classifier per target column.
- Family comparison rule: evaluate Logistic Regression (median impute + scale)
  vs HistGradientBoostingClassifier under the SAME outer CV (5-fold StratifiedKFold,
  computed per-target then averaged). If |gap|/max < 0.005 relative, average their
  test probas; else pick the winner by mean CV.
- skill_tune_when_n_train_supports_it: small 3-config grid for the GBM family.

No NaNs in this dataset; no missing-indicator skill firing. All features numeric/continuous
(image-derived). The skill_stringify_low_cardinality_int_codes precondition fires for a few
low-cardinality numeric columns BUT for the GBM we don't need scaling/OHE; for LR we apply
StandardScaler which would in principle benefit -- skipping stringification because the
columns (Length_of_Conveyer, Steel_Plate_Thickness, X/Y perimeter, etc.) all encode true
physical magnitudes, not category codes; stringifying would just bloat the feature space.
"""
from __future__ import annotations
import os

# IMPORTANT: must set BEFORE importing numpy/sklearn. In this sandbox the default
# OpenMP/BLAS thread count makes HistGradientBoosting ~400x slower than single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

START = time.time()
SEED = 0
N_JOBS = 2
N_TRAIN_CAP = 15000
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e3/data")
HERE = Path(__file__).resolve().parent

TARGETS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


def make_lr() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=SEED)),
    ])


def make_gbm(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def cv_score_multi(estimator_factory, X: np.ndarray, Y: pd.DataFrame, n_splits: int = 5) -> tuple[float, np.ndarray]:
    """
    Per-target StratifiedKFold OOF probabilities; returns (mean_auc, oof_proba [n,T]).
    `estimator_factory` is a zero-arg callable that returns a fresh estimator.
    """
    n, T = X.shape[0], Y.shape[1]
    oof = np.zeros((n, T), dtype=np.float64)
    for t, tgt in enumerate(Y.columns):
        y = Y[tgt].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for tr, va in skf.split(X, y):
            est = estimator_factory()
            est.fit(X[tr], y[tr])
            oof[va, t] = est.predict_proba(X[va])[:, 1]
    aucs = [roc_auc_score(Y.iloc[:, t].values, oof[:, t]) for t in range(T)]
    return float(np.mean(aucs)), oof


def fit_full_predict(estimator_factory, X: np.ndarray, Y: pd.DataFrame, X_test: np.ndarray) -> np.ndarray:
    n_test, T = X_test.shape[0], Y.shape[1]
    P = np.zeros((n_test, T), dtype=np.float64)
    for t, tgt in enumerate(Y.columns):
        est = estimator_factory()
        est.fit(X, Y[tgt].values)
        P[:, t] = est.predict_proba(X_test)[:, 1]
    return P


def main() -> None:
    print(f"[{time.time()-START:5.1f}s] Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    if len(train) > N_TRAIN_CAP:
        train = train.sample(n=N_TRAIN_CAP, random_state=SEED).reset_index(drop=True)
    print(f"  train={train.shape}, test={test.shape}")

    feature_cols = [c for c in train.columns if c not in TARGETS and c != "id"]
    X = train[feature_cols].values.astype(np.float64)
    Y = train[TARGETS].copy()
    X_test = test[feature_cols].values.astype(np.float64)

    audit: dict = {"family_comparison": {}, "hyperparameter_grid": [], "chosen_config": None}

    # --- Family A: Logistic Regression (linear baseline) ---
    print(f"[{time.time()-START:5.1f}s] CV: LogisticRegression (median+scale)...")
    lr_cv, _ = cv_score_multi(make_lr, X, Y, n_splits=5)
    print(f"  LR mean AUC: {lr_cv:.5f}")
    audit["family_comparison"]["logreg"] = lr_cv

    # --- Family B: HistGradientBoosting, small 3-config grid (skill_tune_when_n_train_supports_it) ---
    print(f"[{time.time()-START:5.1f}s] CV: HistGradientBoosting (3-config grid)...")
    grid = [
        dict(max_iter=200, learning_rate=0.05, max_depth=None, max_leaf_nodes=31),
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=63),
        dict(max_iter=400, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=1.0),
    ]
    gbm_results = []
    for params in grid:
        if time.time() - START > 140:
            print(f"  [budget guard] skipping config {params}")
            break
        score, _ = cv_score_multi(lambda p=params: make_gbm(p), X, Y, n_splits=5)
        print(f"  {params}: {score:.5f}")
        gbm_results.append((score, params))
        audit["hyperparameter_grid"].append({"params": params, "cv_auc": score})

    gbm_results.sort(reverse=True, key=lambda x: x[0])
    best_gbm_score, best_gbm_params = gbm_results[0]
    audit["chosen_config"] = best_gbm_params
    audit["family_comparison"]["hgbc"] = best_gbm_score
    print(f"  best GBM config: {best_gbm_params} ({best_gbm_score:.5f})")

    # --- Family-comparison rule (relative gap < 0.005 => average) ---
    s_a, s_b = lr_cv, best_gbm_score
    rel_gap = abs(s_a - s_b) / max(abs(s_a), abs(s_b))
    audit["family_comparison"]["relative_gap"] = rel_gap
    print(f"[{time.time()-START:5.1f}s] LR={s_a:.5f}, GBM={s_b:.5f}, rel_gap={rel_gap:.5f}")

    # --- Fit-full & predict (always need GBM; LR only if averaging or LR wins) ---
    print(f"[{time.time()-START:5.1f}s] Refit on full train, predict test...")
    P_gbm = fit_full_predict(lambda p=best_gbm_params: make_gbm(p), X, Y, X_test)

    if rel_gap < 0.005:
        P_lr = fit_full_predict(make_lr, X, Y, X_test)
        P = 0.5 * (P_gbm + P_lr)
        chosen = "ensemble(LR+HGBC)"
        headline_cv = 0.5 * (s_a + s_b)
    elif s_b >= s_a:
        P = P_gbm
        chosen = "HistGradientBoosting"
        headline_cv = s_b
    else:
        P = fit_full_predict(make_lr, X, Y, X_test)
        chosen = "LogisticRegression"
        headline_cv = s_a

    audit["family_comparison"]["chosen"] = chosen
    print(f"  chosen: {chosen} (mean CV AUC = {headline_cv:.5f})")

    # --- Write submission ---
    sub = pd.DataFrame({"id": test["id"].values})
    for t, tgt in enumerate(TARGETS):
        sub[tgt] = P[:, t]
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[{time.time()-START:5.1f}s] Wrote {sub_path} (shape={sub.shape})")

    # --- result.json ---
    per_target_auc = {t: float(audit["family_comparison"]["hgbc"]) for t in TARGETS}  # placeholder; mean is the headline
    result = {
        "target_column_chosen": TARGETS,
        "headline_metric": "mean_column_roc_auc",
        "headline_value": float(headline_cv),
        "all_metrics": {
            "lr_mean_cv_auc": float(s_a),
            "hgbc_mean_cv_auc": float(s_b),
            "ensemble_used": rel_gap < 0.005,
            "rel_gap": float(rel_gap),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Multi-label 7-target ROC AUC. Compared LogReg(median+scale) vs HistGradientBoosting "
            "under same 5-fold StratifiedKFold per target. Applied the family-comparison "
            "averaging rule (rel_gap<0.005 => 50/50 ensemble of test probas)."
        ),
        "audit": audit,
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{time.time()-START:5.1f}s] Wrote result.json")
    print(f"[{time.time()-START:5.1f}s] DONE")


if __name__ == "__main__":
    main()
