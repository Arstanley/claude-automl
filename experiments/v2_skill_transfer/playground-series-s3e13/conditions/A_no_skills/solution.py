"""
Playground Series S3E13 — Vector Borne Disease Prediction.

Multi-class classification (11 diseases) from 64 binary symptom features.
Evaluation: MPA@3 — top-3 predicted labels per row, scored 1/(rank) style
(equivalent to Mean Average Precision @ 3 when there's exactly one true label
per row; we use MAP@3 as the implementation).

Approach:
  1. Load train/test (n_train=565, n_test=142, 11 classes).
  2. Build an ensemble: LogisticRegression + RandomForest + GaussianNB, soft-vote on
     probabilities.  Each model is reasonable for small-n binary-feature data.
  3. Stratified 5-fold CV to estimate MAP@3 (== MPA@3 in the spec).
  4. Refit on full train, take top-3 class probabilities for each test row,
     emit space-separated labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
SUBMISSION_CSV = HERE / "submission.csv"
RESULT_JSON = HERE / "result.json"

TARGET = "prognosis"
ID_COL = "id"
RNG = 42


def map_at_k(y_true: np.ndarray, top_k_preds: np.ndarray, k: int = 3) -> float:
    """Mean Average Precision at K when each row has exactly one true label.

    Score per row = 1/rank if true label is in top-k else 0.
    """
    n = len(y_true)
    score = 0.0
    for i in range(n):
        row = top_k_preds[i, :k]
        for r, p in enumerate(row, start=1):
            if p == y_true[i]:
                score += 1.0 / r
                break
    return score / n


def hit_at_k(y_true: np.ndarray, top_k_preds: np.ndarray, k: int = 3) -> float:
    """Fraction of rows where the true label appears in top-k (1/0 hit rate)."""
    n = len(y_true)
    hits = 0
    for i in range(n):
        if y_true[i] in top_k_preds[i, :k]:
            hits += 1
    return hits / n


def top_k_from_proba(proba: np.ndarray, classes: np.ndarray, k: int = 3) -> np.ndarray:
    """Return an (n, k) array of class labels ordered by descending probability."""
    # argsort ascending, take last k, then reverse for descending
    order = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
    return classes[order]


def build_ensemble(seed: int = RNG):
    """Returns a list of (name, estimator) pairs that we soft-vote on probabilities."""
    return [
        (
            "logreg",
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ("bnb", BernoulliNB(alpha=1.0)),
    ]


def fit_ensemble(X: np.ndarray, y: np.ndarray, seed: int = RNG):
    models = build_ensemble(seed)
    fitted = []
    for name, est in models:
        est.fit(X, y)
        fitted.append((name, est))
    return fitted


def predict_proba_ensemble(fitted, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average predicted probabilities across the ensemble.

    Returns (proba, classes_) — all members share the same classes_ since they
    were fit on the same y.
    """
    probas = []
    classes_ref = fitted[0][1].classes_
    for name, est in fitted:
        p = est.predict_proba(X)
        # Align columns just in case (should already match given identical y).
        if not np.array_equal(est.classes_, classes_ref):
            # Reorder p so columns match classes_ref
            idx = [list(est.classes_).index(c) for c in classes_ref]
            p = p[:, idx]
        probas.append(p)
    return np.mean(probas, axis=0), classes_ref


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X = train[feature_cols].values.astype(np.float32)
    y_str = train[TARGET].values

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes_ = le.classes_

    print(f"train={X.shape}, test={test.shape}, n_classes={len(classes_)}")
    print("classes:", list(classes_))

    # -------- Cross-validated MAP@3 / hit@3 --------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    cv_map3, cv_hit3, cv_top1_acc = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        fitted = fit_ensemble(X[tr_idx], y[tr_idx])
        proba, ens_classes = predict_proba_ensemble(fitted, X[va_idx])
        top3 = top_k_from_proba(proba, ens_classes, k=3)
        map3 = map_at_k(y[va_idx], top3, k=3)
        hit3 = hit_at_k(y[va_idx], top3, k=3)
        top1 = (top3[:, 0] == y[va_idx]).mean()
        print(f"fold {fold}: MAP@3={map3:.4f}  hit@3={hit3:.4f}  top1_acc={top1:.4f}")
        cv_map3.append(map3)
        cv_hit3.append(hit3)
        cv_top1_acc.append(top1)

    mean_map3 = float(np.mean(cv_map3))
    std_map3 = float(np.std(cv_map3))
    mean_hit3 = float(np.mean(cv_hit3))
    mean_top1 = float(np.mean(cv_top1_acc))
    print(f"CV MAP@3 = {mean_map3:.4f} ± {std_map3:.4f}")
    print(f"CV hit@3 = {mean_hit3:.4f}")
    print(f"CV top1 acc = {mean_top1:.4f}")

    # -------- Refit on all train, predict test --------
    fitted_full = fit_ensemble(X, y)
    X_test = test[feature_cols].values.astype(np.float32)
    proba_test, ens_classes = predict_proba_ensemble(fitted_full, X_test)
    top3_test_enc = top_k_from_proba(proba_test, ens_classes, k=3)
    # Decode integer class indices back to label strings.
    top3_test = le.inverse_transform(top3_test_enc.ravel()).reshape(top3_test_enc.shape)

    # Format: "label1 label2 label3" (underscores already present in labels).
    pred_strings = [" ".join(row) for row in top3_test]
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred_strings})
    sub.to_csv(SUBMISSION_CSV, index=False)
    print(f"Wrote {SUBMISSION_CSV} ({len(sub)} rows)")

    # -------- result.json --------
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "MAP@3 (== MPA@3 per spec)",
        "headline_value": mean_map3,
        "all_metrics": {
            "cv_map_at_3_mean": mean_map3,
            "cv_map_at_3_std": std_map3,
            "cv_map_at_3_per_fold": [float(x) for x in cv_map3],
            "cv_hit_at_3_mean": mean_hit3,
            "cv_top1_accuracy_mean": mean_top1,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_classes": int(len(classes_)),
            "n_features": int(len(feature_cols)),
        },
        "interpretation_notes": (
            "11-class disease classification on 64 binary symptom features (n_train=565, "
            "n_test=142). Headline metric MAP@3 is the spec's MPA@3: per-row 1/rank of the "
            "true label if it lies in the top-3 predicted labels, else 0; averaged across "
            "rows. Submitting 3 labels per row maximises expected MPA@3, so we do that. "
            "Model is a soft-vote ensemble of LogisticRegression + RandomForest(600) + "
            "BernoulliNB on the binary features; CV is 5-fold stratified on the label. "
            "Hit@3 (label in top-3 ignoring rank) and top-1 accuracy are reported alongside "
            "for context."
        ),
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"Wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()
