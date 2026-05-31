"""Playground Series S3E3 — Employee Attrition prediction.

Task: predict probability of `Attrition` (binary 0/1) on a synthetic
tabular dataset. Evaluation metric is ROC AUC.

This script:
  1. Loads train/test CSVs from ../data/.
  2. Builds a sklearn pipeline (one-hot for categoricals, passthrough
     for numerics) + LogisticRegression and GradientBoostingClassifier.
  3. Picks the model with the best CV AUC.
  4. Refits on full train, predicts probabilities on test.
  5. Writes submission.csv (id, Attrition) and result.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

ID_COL = "id"
TARGET = "Attrition"
RANDOM_STATE = 42


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    # Drop constant columns (e.g. EmployeeCount, StandardHours, Over18) — they
    # add no signal and just bloat the one-hot space.
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=False) > 1]
    num_cols = [c for c in num_cols if df[c].nunique(dropna=False) > 1]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"train: {train.shape}  test: {test.shape}")
    print(f"target balance:\n{train[TARGET].value_counts(normalize=True)}")

    feature_cols = [c for c in train.columns if c not in {ID_COL, TARGET}]
    pre = build_preprocessor(train, feature_cols)

    candidates: dict[str, Pipeline] = {
        "logreg": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gbm": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    X = train[feature_cols]
    y = train[TARGET].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores: dict[str, float] = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_scores[name] = float(np.mean(scores))
        print(f"  CV AUC {name:7s}: mean={scores.mean():.4f}  std={scores.std():.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    best_auc = cv_scores[best_name]
    print(f"best model: {best_name}  (CV AUC = {best_auc:.4f})")

    best_pipe = candidates[best_name].fit(X, y)
    test_X = test[feature_cols]
    proba = best_pipe.predict_proba(test_X)[:, 1]

    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: proba})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  ({len(sub)} rows)")
    print(sub.head())

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "roc_auc",
        "headline_value": best_auc,
        "all_metrics": {
            "cv_roc_auc_logreg": cv_scores["logreg"],
            "cv_roc_auc_gbm": cv_scores["gbm"],
            "cv_folds": 5,
            "selected_model": best_name,
        },
        "interpretation_notes": (
            "Binary classification of employee Attrition (0/1). Target is "
            "imbalanced (~84% no-attrition). Headline metric is 5-fold "
            "stratified CV ROC AUC; we report both candidates and submit "
            "probabilities from the best one refit on the full training set. "
            "ID column in the data is `id` (not `EmployeeNumber` as the "
            "spec's example suggested)."
        ),
    }
    result_path = HERE / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
