#!/usr/bin/env python3
"""
Playground Series S3E5 — Wine Quality (QWK)
Solver under the ambiguous spec, addressing the Challenger audit.

Design (audit-driven):
  * Target column: `quality` (only column in train missing from test). Integers in {3..8}.
  * Submission Ids come from test.csv (NOT sample_submission.csv, whose Ids 2056-3427
    are disjoint from test). Row order is preserved.
  * QWK is optimized via the standard regression-with-threshold-tuning recipe:
      - Train ordinal regressors (target as float).
      - Average OOF predictions across an ensemble of models.
      - Fit 5 monotone thresholds on OOF predictions to maximize QWK directly.
      - Apply thresholds to the test-set continuous predictions, clip to [3,8].
  * Clip output to {3..8}; class 3 (n=10) and class 8 (n=33) are the high-leverage
    classes per the audit's distance-penalty analysis.
  * Stratified K-fold with K=5 (class 3 has 10 rows → 2 per fold).
  * No external data, no internet, no GPU. Sklearn / pandas / numpy only.
  * Id is excluded from features (audit: Id is a non-informative leakage proxy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

# ---------------- paths -----------------
HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
SUB_CSV = HERE / "submission.csv"
RESULT_JSON = HERE / "result.json"

RNG = 42
N_CLASSES_EVAL = 10  # eval.py hardcodes N=10; we report QWK under that convention.
LABEL_LO, LABEL_HI = 3, 8  # observed range in train; clip predictions accordingly.


def qwk(y_true, y_pred, N=N_CLASSES_EVAL):
    """QWK matching the hidden evaluator: integer labels in [0, N-1]."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    # Use sklearn with explicit label list so it matches eval.py's (N,N) confusion matrix.
    labels = list(range(N))
    return cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")


def apply_thresholds(cont, thresholds):
    """Given monotone thresholds t1<t2<t3<t4<t5, map continuous → {3..8}."""
    # 5 thresholds split into 6 bins corresponding to classes 3..8.
    t = np.sort(thresholds)
    out = np.full_like(cont, LABEL_LO, dtype=int)
    for i, ti in enumerate(t):
        out[cont >= ti] = LABEL_LO + i + 1
    return out


def optimize_thresholds(oof_cont, y_true):
    """Fit 5 thresholds on OOF predictions to maximize QWK."""
    # Initial guess: midpoints between adjacent class means.
    init = np.array([3.5, 4.5, 5.5, 6.5, 7.5], dtype=float)

    def neg_qwk(t):
        preds = apply_thresholds(oof_cont, t)
        return -qwk(y_true, preds)

    best = init
    best_score = -neg_qwk(init)
    # Nelder-Mead from a few seeds for robustness.
    for seed_offset in [np.zeros(5), np.array([-.2, -.1, 0, .1, .2]),
                        np.array([.1, .1, .1, .1, .1]),
                        np.array([-.1, -.1, -.1, -.1, -.1])]:
        x0 = init + seed_offset
        res = minimize(neg_qwk, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        if -res.fun > best_score:
            best_score = -res.fun
            best = res.x
    return np.sort(best), best_score


def make_models(seed):
    """Diverse ordinal regressors. Trees handle the differing feature scales fine."""
    return {
        "gbr": GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.85, random_state=seed,
        ),
        "rf": RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            max_features=0.7, n_jobs=-1, random_state=seed,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            max_features=0.7, n_jobs=-1, random_state=seed,
        ),
    }


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Sanity vs. audit ground facts.
    assert "quality" in train.columns and "quality" not in test.columns, \
        "Target column must be `quality` (only column missing from test)."
    feature_cols = [c for c in train.columns if c not in ("Id", "quality")]
    assert len(feature_cols) == 11, f"Expected 11 features, got {len(feature_cols)}"
    print(f"[data] train={len(train)} rows, test={len(test)} rows, features={feature_cols}")
    print(f"[data] target distribution:\n{train['quality'].value_counts().sort_index()}")

    X = train[feature_cols].values.astype(np.float64)
    y = train["quality"].values.astype(np.int64)
    X_test = test[feature_cols].values.astype(np.float64)

    # K=5 stratified: class 3 has 10 rows → 2 per fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

    oof_cont = np.zeros(len(train), dtype=np.float64)
    test_cont = np.zeros(len(test), dtype=np.float64)
    n_folds = 0

    per_model_oof = {m: np.zeros(len(train)) for m in make_models(0)}
    per_model_test = {m: np.zeros(len(test)) for m in make_models(0)}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        n_folds += 1
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx].astype(float), y[va_idx]
        fold_models = make_models(RNG + fold)
        fold_va = np.zeros(len(va_idx))
        fold_te = np.zeros(len(test))
        for name, mdl in fold_models.items():
            mdl.fit(Xtr, ytr)
            va_pred = mdl.predict(Xva)
            te_pred = mdl.predict(X_test)
            per_model_oof[name][va_idx] = va_pred
            per_model_test[name] += te_pred / skf.n_splits
            fold_va += va_pred / len(fold_models)
            fold_te += te_pred / len(fold_models)
        oof_cont[va_idx] = fold_va
        test_cont += fold_te / skf.n_splits
        # Per-fold reasonableness check: round-and-clip QWK
        rounded = np.clip(np.round(fold_va).astype(int), LABEL_LO, LABEL_HI)
        print(f"[fold {fold}] naive-round QWK={qwk(yva, rounded):.4f}  "
              f"(cont range {fold_va.min():.2f}..{fold_va.max():.2f})")

    # Per-model OOF QWK with naive round, for diagnostics.
    per_model_oof_qwk = {}
    for name, p in per_model_oof.items():
        rounded = np.clip(np.round(p).astype(int), LABEL_LO, LABEL_HI)
        per_model_oof_qwk[name] = float(qwk(y, rounded))
    print(f"[diag] per-model naive-round OOF QWK: {per_model_oof_qwk}")

    # Naive baseline metrics on the OOF ensemble.
    naive_oof_pred = np.clip(np.round(oof_cont).astype(int), LABEL_LO, LABEL_HI)
    naive_oof_qwk = float(qwk(y, naive_oof_pred))
    print(f"[oof] ensemble naive-round QWK={naive_oof_qwk:.4f}")

    # Threshold optimization on OOF (legitimate; no test peek).
    thresholds, tuned_oof_qwk = optimize_thresholds(oof_cont, y)
    tuned_oof_qwk = float(tuned_oof_qwk)
    print(f"[oof] tuned thresholds={thresholds.round(4).tolist()}  QWK={tuned_oof_qwk:.4f}")

    # Sanity baselines (audit says constant predictor must score QWK = 0.0).
    const_qwks = {f"const_{c}": float(qwk(y, np.full_like(y, c))) for c in (5, 6, 7)}
    print(f"[oof] constant-prediction sanity QWKs (should be ~0): {const_qwks}")

    # Build final test predictions.
    test_pred = apply_thresholds(test_cont, thresholds)
    test_pred = np.clip(test_pred, LABEL_LO, LABEL_HI).astype(int)
    print(f"[test] prediction distribution: {pd.Series(test_pred).value_counts().sort_index().to_dict()}")

    # Submission: Ids from test.csv in test.csv row order. Header = 'Id,quality'.
    sub = pd.DataFrame({"Id": test["Id"].values, "quality": test_pred})
    assert len(sub) == len(test) == 412, f"Submission row count mismatch: {len(sub)}"
    assert list(sub.columns) == ["Id", "quality"]
    assert sub["quality"].between(LABEL_LO, LABEL_HI).all()
    assert sub["quality"].dtype.kind == "i"
    sub.to_csv(SUB_CSV, index=False)
    print(f"[write] {SUB_CSV} ({len(sub)} rows)")

    # Confirm no sample_submission Id contamination.
    samp_ids = set(pd.read_csv(DATA / "sample_submission.csv")["Id"].tolist())
    test_ids = set(test["Id"].tolist())
    overlap = samp_ids & test_ids
    print(f"[check] sample_submission ∩ test Ids = {len(overlap)} (audit says 0)")

    result = {
        "target_column_chosen": "quality",
        "target_column_justification": (
            "The only column present in train.csv but absent from test.csv is `quality`; "
            "spec uses placeholder `<target>` so we identify the target by schema diff. "
            "Target is integer-valued in {3..8} per train; predictions cast to int and clipped "
            "to [3,8] (the audit's recommended range) to avoid the IndexError trap in eval.py's "
            "(10,10) confusion matrix."
        ),
        "headline_metric": "quadratic_weighted_kappa_oof",
        "headline_value": tuned_oof_qwk,
        "all_metrics": {
            "oof_qwk_naive_round": naive_oof_qwk,
            "oof_qwk_threshold_tuned": tuned_oof_qwk,
            "per_model_oof_qwk_naive_round": per_model_oof_qwk,
            "constant_prediction_qwk_sanity": const_qwks,
            "tuned_thresholds": thresholds.tolist(),
            "n_folds": n_folds,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_prediction_distribution": {
                int(k): int(v) for k, v in
                pd.Series(test_pred).value_counts().sort_index().to_dict().items()
            },
            "train_class_distribution": {
                int(k): int(v) for k, v in
                train["quality"].value_counts().sort_index().to_dict().items()
            },
        },
        "audit_items_addressed": [
            "Target identified as `quality` via schema diff (placeholder `<target>` in spec).",
            "Submission Ids drawn from test.csv (412 rows, ids 23-2048) — NOT from sample_submission.csv (1372 rows, ids 2056-3427, disjoint).",
            "Submission row order matches test.csv row order; no sort/reorder applied.",
            "Header is exactly `Id,quality` to satisfy the eval's column lookup.",
            "Predictions are integers, clipped to [3,8] (audit-recommended, also safe under the [0,9] requirement).",
            "QWK optimized via regression + monotone threshold tuning on OOF (the audited correct recipe), NOT naive multiclass argmax.",
            "Stratified 5-fold CV: K=5 chosen so class 3 (n=10) places 2 rows per fold.",
            "Thresholds tuned on OOF only — no test-set leakage.",
            "Id excluded from features (audit flags Id-based features as leakage proxy).",
            "Constant-prediction QWK sanity reported to confirm we beat the QWK=0 chance baseline meaningfully.",
            "No external data, no internet, no GPU; sklearn/pandas/numpy only.",
        ],
        "interpretation_notes": (
            "QWK is computed with N=10 to match eval.py exactly (cohen_kappa_score with "
            "labels=range(10) and weights='quadratic'). Headline value is OOF QWK after fitting "
            "5 monotone thresholds on OOF continuous predictions of a GBR+RF+ET ensemble. "
            "Constant predictors verifiably score ~0.0 QWK on OOF, confirming the baseline. "
            "Naive round-and-clip without threshold tuning is reported for comparison; "
            "threshold tuning is the standard QWK uplift on this ordinal target and is "
            "explicitly endorsed by the audit. Test predictions land in {%d..%d}; no value "
            "is outside the eval matrix's [0,9] range so eval.py cannot IndexError."
            % (LABEL_LO, LABEL_HI)
        ),
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {RESULT_JSON}")
    print(f"[summary] headline OOF QWK (threshold-tuned) = {tuned_oof_qwk:.4f}")


if __name__ == "__main__":
    main()
