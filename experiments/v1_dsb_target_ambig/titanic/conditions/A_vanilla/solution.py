"""Titanic survival prediction — vanilla baseline.

Target: Survived (binary classification)
Metric: accuracy

Pipeline: simple feature engineering + GradientBoostingClassifier with
a small CV grid search. Holdout accuracy is reported in result.json.
"""
from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_CSV = os.path.join(HERE, "submission.csv")
RESULT_JSON = os.path.join(HERE, "result.json")

TARGET = "Survived"
ID_COL = "PassengerId"

TITLE_RE = re.compile(r",\s*([^.]+)\.")


def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    m = TITLE_RE.search(name)
    if not m:
        return "Unknown"
    title = m.group(1).strip()
    # Collapse rare titles.
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    if title in {"Mr", "Mrs", "Miss", "Master"}:
        return title
    return "Rare"


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].map(extract_title)
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["CabinDeck"] = df["Cabin"].astype(str).str[0].where(df["Cabin"].notna(), "U")
    df["TicketLen"] = df["Ticket"].astype(str).str.len()
    # Keep only modeling columns.
    return df[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title",
            "FamilySize",
            "IsAlone",
            "HasCabin",
            "CabinDeck",
            "TicketLen",
        ]
    ]


def build_pipeline() -> Pipeline:
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin", "TicketLen"]
    categorical = ["Sex", "Embarked", "Title", "CabinDeck"]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", model)])


def main() -> None:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    y = train_df[TARGET].astype(int).values
    X = featurize(train_df)
    X_test = featurize(test_df)

    pipe = build_pipeline()

    # 5-fold CV on the full training set.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    # Holdout split for a single-number "headline" eval.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe_holdout = build_pipeline()
    pipe_holdout.fit(X_tr, y_tr)
    val_pred = pipe_holdout.predict(X_val)
    val_acc = float((val_pred == y_val).mean())

    # Refit on all of train and predict on test.
    pipe.fit(X, y)
    test_pred = pipe.predict(X_test).astype(int)

    submission = pd.DataFrame(
        {ID_COL: test_df[ID_COL].values, TARGET: test_pred}
    )
    submission.to_csv(SUBMISSION_CSV, index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "accuracy",
        "headline_value": cv_mean,
        "all_metrics": {
            "cv5_accuracy_mean": cv_mean,
            "cv5_accuracy_std": cv_std,
            "cv5_accuracy_folds": [float(s) for s in cv_scores],
            "holdout_accuracy": val_acc,
        },
        "interpretation_notes": (
            "Target column is unambiguous: Survived (the only column in train.csv "
            "but not test.csv) with accuracy as the stated metric. Used standard "
            "Titanic feature engineering (Title from Name, FamilySize, IsAlone, "
            "HasCabin, CabinDeck) plus a GradientBoostingClassifier; headline value "
            "is mean 5-fold stratified CV accuracy."
        ),
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"CV5 accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"Holdout accuracy: {val_acc:.4f}")
    print(f"Wrote {SUBMISSION_CSV} ({len(submission)} rows)")
    print(f"Wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()
