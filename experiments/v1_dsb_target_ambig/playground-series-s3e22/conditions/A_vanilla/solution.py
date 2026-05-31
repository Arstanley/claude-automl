"""
Vanilla baseline for Kaggle Playground Series S3E22 (Horse Survival).

Target: `outcome` (multi-class: lived / died / euthanized)
Evaluation: micro-averaged F1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
SEED = 42

TARGET = "outcome"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )


def evaluate_cv(pipe, X, y, name: str) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_micro", n_jobs=1)
    mean = float(scores.mean())
    std = float(scores.std())
    print(f"  {name}: micro-F1 = {mean:.4f} +/- {std:.4f}")
    return mean


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    print(f"train: {train.shape}  test: {test.shape}")
    print(f"target distribution: {train[TARGET].value_counts().to_dict()}")

    feature_cols = [c for c in train.columns if c not in {"id", TARGET}]
    # hospital_number is an ID-like high-cardinality int — drop it.
    if "hospital_number" in feature_cols:
        feature_cols.remove("hospital_number")

    X = train[feature_cols].copy()
    y = train[TARGET].astype(str).values
    X_test = test[feature_cols].copy()

    preprocessor = build_preprocessor(X)

    candidates = {
        "logreg": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
                ),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "gbm": Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    print("\n5-fold stratified CV (micro-F1):")
    cv_scores: dict[str, float] = {}
    for name, pipe in candidates.items():
        cv_scores[name] = evaluate_cv(pipe, X, y, name)

    best_name = max(cv_scores, key=cv_scores.get)
    best_score = cv_scores[best_name]
    print(f"\nBest model by CV: {best_name} (micro-F1 = {best_score:.4f})")

    best_pipe = candidates[best_name]
    best_pipe.fit(X, y)

    # in-sample train micro-F1 (sanity)
    train_pred = best_pipe.predict(X)
    train_f1 = float(f1_score(y, train_pred, average="micro"))
    print(f"in-sample micro-F1 (informational): {train_f1:.4f}")

    preds = best_pipe.predict(X_test)
    sub = pd.DataFrame({"id": test["id"].values, TARGET: preds})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  rows={len(sub)}")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "micro_f1_cv5",
        "headline_value": best_score,
        "all_metrics": {
            "cv_micro_f1_per_model": cv_scores,
            "best_model": best_name,
            "train_micro_f1_insample": train_f1,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_classes": int(len(np.unique(y))),
            "classes": sorted(np.unique(y).tolist()),
        },
        "interpretation_notes": (
            "Three-class horse survival classification (lived/died/euthanized). "
            "Target is `outcome` per the spec. Submission uses raw class labels in "
            "the original string form (matching sample_submission.csv). "
            "Headline metric is 5-fold stratified CV micro-F1 of the best model "
            f"selected by CV ({best_name}). hospital_number was dropped as an "
            "ID-like high-cardinality column. Missing values are imputed "
            "(median for numeric, most_frequent for categorical), categoricals "
            "are one-hot encoded, numerics are standardized."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {HERE / 'result.json'}")


if __name__ == "__main__":
    main()
