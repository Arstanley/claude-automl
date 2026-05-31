"""
Solution for Kaggle "Real or Not? NLP with Disaster Tweets" (binary text classification).

Approach:
- TF-IDF features on tweet text (word + char_wb n-grams), plus tiny categorical features
  derived from `keyword` and missingness indicators for `keyword` / `location`.
- Logistic Regression with a small explicit C-grid tuned via 5-fold StratifiedKFold CV on F1.
- Refit best config on full train and predict on test.

Metric: F1 (binary). Output: submission.csv (id,target) and result.json.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/nlp-getting-started/data")
RNG = 0


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # normalize URLs and user mentions, collapse whitespace
    s = re.sub(r"https?://\S+", " URL ", s)
    s = re.sub(r"@\w+", " USER ", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Build TF-IDF + small structured features. Returns (X_train, X_test, y)."""
    tr_text = train["text"].map(clean_text)
    te_text = test["text"].map(clean_text)

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
    )

    Xw_tr = word_vec.fit_transform(tr_text)
    Xw_te = word_vec.transform(te_text)
    Xc_tr = char_vec.fit_transform(tr_text)
    Xc_te = char_vec.transform(te_text)

    # Keyword as one-hot (low cardinality ~220); missing -> "__NA__"
    kw_tr = train["keyword"].fillna("__NA__").astype(str).values.reshape(-1, 1)
    kw_te = test["keyword"].fillna("__NA__").astype(str).values.reshape(-1, 1)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    Xk_tr = ohe.fit_transform(kw_tr)
    Xk_te = ohe.transform(kw_te)

    # Missingness indicators (location is ~33% NaN — non-trivial signal)
    miss_tr = np.column_stack([
        train["keyword"].isna().astype(np.float32).values,
        train["location"].isna().astype(np.float32).values,
    ])
    miss_te = np.column_stack([
        test["keyword"].isna().astype(np.float32).values,
        test["location"].isna().astype(np.float32).values,
    ])
    Xm_tr = sparse.csr_matrix(miss_tr)
    Xm_te = sparse.csr_matrix(miss_te)

    X_tr = sparse.hstack([Xw_tr, Xc_tr, Xk_tr, Xm_tr]).tocsr()
    X_te = sparse.hstack([Xw_te, Xc_te, Xk_te, Xm_te]).tocsr()
    return X_tr, X_te


def cv_f1(X, y, C: float) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        clf = LogisticRegression(
            C=C, solver="liblinear", max_iter=2000, random_state=RNG,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        scores.append(f1_score(y[va_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  train: {train.shape}  test: {test.shape}")
    print(f"  target dist: {train['target'].value_counts().to_dict()}")

    print("Building features...")
    X_tr, X_te = build_features(train, test)
    y = train["target"].astype(int).values
    print(f"  X_tr nnz/shape: {X_tr.nnz}/{X_tr.shape}")

    # Small explicit C grid for LogReg
    print("Tuning C via 5-fold CV (F1)...")
    grid = [0.3, 1.0, 3.0, 10.0]
    cv_results = {}
    best = (-1.0, None)
    for C in grid:
        m, s = cv_f1(X_tr, y, C)
        cv_results[str(C)] = {"mean_f1": m, "std_f1": s}
        print(f"  C={C}: F1={m:.4f} ± {s:.4f}")
        if m > best[0]:
            best = (m, C)
    best_C = best[1]
    print(f"  best C={best_C}, mean F1={best[0]:.4f}")

    # Refit on full train and predict
    print("Refitting on full train and predicting...")
    final = LogisticRegression(
        C=best_C, solver="liblinear", max_iter=4000, random_state=RNG,
    )
    final.fit(X_tr, y)

    # OOF predictions to recompute headline metric with the chosen C
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in cv.split(X_tr, y):
        clf = LogisticRegression(
            C=best_C, solver="liblinear", max_iter=2000, random_state=RNG,
        )
        clf.fit(X_tr[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X_tr[va_idx])
    headline_f1 = float(f1_score(y, oof))
    all_metrics = {
        "f1_oof": headline_f1,
        "precision_oof": float(precision_score(y, oof)),
        "recall_oof": float(recall_score(y, oof)),
        "accuracy_oof": float(accuracy_score(y, oof)),
        "cv_grid": cv_results,
    }
    print(f"  OOF F1={headline_f1:.4f}  P={all_metrics['precision_oof']:.4f}  R={all_metrics['recall_oof']:.4f}")

    test_pred = final.predict(X_te).astype(int)
    sub = pd.DataFrame({"id": test["id"].astype(int).values, "target": test_pred})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  wrote {sub_path}  ({len(sub)} rows)")

    result = {
        "target_column_chosen": "target",
        "headline_metric": "f1",
        "headline_value": headline_f1,
        "all_metrics": all_metrics,
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_missing_indicator_when_missingness_signals_class.md",
        ],
        "interpretation_notes": (
            "TF-IDF (word 1-2grams + char_wb 3-5grams) + one-hot keyword + "
            "missingness indicators for keyword/location, fed to LogisticRegression. "
            "Tuned C via small 4-config 5-fold StratifiedKFold grid on F1; refit on full train. "
            f"Chose C={best_C}. Linear models on TF-IDF are the canonical strong baseline for short-text "
            "disaster-tweet classification."
        ),
        "best_C": best_C,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    result_path = HERE / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {result_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
