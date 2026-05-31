"""
Solution for Tabular Playground Series S3E5 (wine quality).

Task: Predict integer `quality` for each row in test.csv. Evaluated by Quadratic
Weighted Kappa (QWK). We train several classifiers/regressors with stratified
cross-validation, score each by QWK on out-of-fold predictions, pick the best,
then refit on full train and write predictions to submission.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TARGET = "quality"
ID_COL = "Id"
RANDOM_STATE = 42
N_SPLITS = 5


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa over integer ordinal labels."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def round_clip(preds: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Round regression outputs to nearest int and clip to ordinal range."""
    return np.clip(np.rint(preds).astype(int), lo, hi)


def cv_score_classifier(model, X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y)
    for tr, va in skf.split(X, y):
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
    return qwk(y, oof)


def cv_score_regressor(model, X: np.ndarray, y: np.ndarray, lo: int, hi: int) -> float:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(X, y):
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
    return qwk(y, round_clip(oof, lo, hi))


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X_train = train[feature_cols].astype(float).values
    y_train = train[TARGET].astype(int).values
    X_test = test[feature_cols].astype(float).values
    test_ids = test[ID_COL].values

    classes = np.sort(np.unique(y_train))
    lo, hi = int(classes.min()), int(classes.max())
    print(f"[info] train={len(train)} test={len(test)} features={len(feature_cols)} "
          f"target='{TARGET}' classes={list(classes)}")

    # Candidate models (mix of classifiers + regressors-rounded). Classifiers
    # directly predict ordinal labels; regressors capture ordinal structure and
    # we round predictions back to ints — often the strongest approach for QWK.
    candidates: dict[str, tuple[str, object]] = {
        "logreg_cls": (
            "cls",
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                 random_state=RANDOM_STATE))]),
        ),
        "rf_cls": (
            "cls",
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=RANDOM_STATE),
        ),
        "hgb_cls": (
            "cls",
            HistGradientBoostingClassifier(max_depth=None, max_iter=300,
                                           learning_rate=0.05,
                                           random_state=RANDOM_STATE),
        ),
        "ridge_reg": (
            "reg",
            Pipeline([("scaler", StandardScaler()),
                      ("reg", Ridge(alpha=1.0))]),
        ),
        "rf_reg": (
            "reg",
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=-1, random_state=RANDOM_STATE),
        ),
        "hgb_reg": (
            "reg",
            HistGradientBoostingRegressor(max_depth=None, max_iter=400,
                                          learning_rate=0.05,
                                          random_state=RANDOM_STATE),
        ),
    }

    scores: dict[str, float] = {}
    for name, (kind, model) in candidates.items():
        if kind == "cls":
            s = cv_score_classifier(model, X_train, y_train, classes)
        else:
            s = cv_score_regressor(model, X_train, y_train, lo, hi)
        scores[name] = s
        print(f"[cv] {name:12s} ({kind}) QWK={s:.4f}")

    best_name = max(scores, key=scores.get)
    best_kind, best_model = candidates[best_name]
    best_score = scores[best_name]
    print(f"[best] {best_name} ({best_kind}) QWK={best_score:.4f}")

    # Refit best model on all training data and predict test.
    best_model.fit(X_train, y_train)
    if best_kind == "cls":
        test_pred = best_model.predict(X_test).astype(int)
    else:
        test_pred = round_clip(best_model.predict(X_test), lo, hi)

    # Out-of-fold confusion for interpretation.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y_train)
    # Re-instantiate to avoid state leakage from the final refit.
    _, fresh = candidates[best_name]
    for tr, va in skf.split(X_train, y_train):
        fresh.fit(X_train[tr], y_train[tr])
        pv = fresh.predict(X_train[va])
        if best_kind == "reg":
            pv = round_clip(pv, lo, hi)
        oof[va] = pv.astype(int)
    cm = confusion_matrix(y_train, oof, labels=list(classes)).tolist()
    oof_acc = float((oof == y_train).mean())
    oof_qwk = qwk(y_train, oof)

    # Write submission.
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
    submission_path = HERE / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"[write] {submission_path} rows={len(submission)}")

    # Write result.json.
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "quadratic_weighted_kappa",
        "headline_value": round(best_score, 6),
        "all_metrics": {
            "cv_qwk_by_model": {k: round(v, 6) for k, v in scores.items()},
            "best_model": best_name,
            "best_model_kind": best_kind,
            "oof_qwk_best": round(oof_qwk, 6),
            "oof_accuracy_best": round(oof_acc, 6),
            "oof_confusion_matrix_classes": [int(c) for c in classes],
            "oof_confusion_matrix": cm,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(len(feature_cols)),
            "n_splits": N_SPLITS,
        },
        "interpretation_notes": (
            f"Target is the ordinal wine 'quality' score (range {lo}-{hi}). "
            "Evaluated by Quadratic Weighted Kappa over 5-fold stratified CV. "
            "Tried logistic regression, random forest, gradient boosting, and "
            "histogram-GBM as both classifiers and regressors (regressor "
            f"predictions rounded+clipped to [{lo},{hi}]). Best model was "
            f"'{best_name}' with CV QWK={best_score:.4f}. Class distribution is "
            "imbalanced (mostly 5-6), so QWK is more informative than accuracy: "
            "near-miss predictions (e.g. 5 vs 6) cost much less than far ones "
            "(e.g. 5 vs 8). Final submission uses the best model refit on the "
            "full training set."
        ),
    }
    result_path = HERE / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {result_path}")


if __name__ == "__main__":
    main()
