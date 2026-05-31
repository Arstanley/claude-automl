"""Spaceship Titanic — vanilla baseline.

Target: `Transported` (boolean). Metric: classification accuracy.

Pipeline:
  - Parse Cabin -> Deck/Num/Side, PassengerId -> Group/GroupSize.
  - Feature engineer total spend & no-spend flag (cryosleep ~ no spend).
  - Impute (median for numeric, most_frequent for categorical) then one-hot encode.
  - Gradient Boosting classifier; 5-fold CV for headline metric estimate.
  - Refit on full train, predict on test, write submission.csv + result.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Cabin -> Deck/Num/Side
    cabin = df["Cabin"].astype("string").str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]
    # PassengerId -> Group / GroupSize
    pid = df["PassengerId"].astype("string").str.split("_", expand=True)
    df["Group"] = pd.to_numeric(pid[0], errors="coerce")
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    # Spend features
    spend = df[SPEND_COLS].fillna(0.0)
    df["TotalSpend"] = spend.sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    # Drop raw text we don't model
    df = df.drop(columns=["Cabin", "Name", "PassengerId"])
    # Normalize categorical-ish cols to plain object with np.nan for missing
    # (pandas NA in StringDtype/BooleanDtype breaks SimpleImputer in older sklearn).
    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]:
        s = df[col]
        df[col] = s.astype(object).where(s.notna(), np.nan).map(
            lambda v: v if v is np.nan or (isinstance(v, float) and np.isnan(v)) else str(v)
        )
    return df


def build_pipeline() -> Pipeline:
    numeric = ["Age", "CabinNum", "Group", "GroupSize", "TotalSpend", "NoSpend",
               *SPEND_COLS]
    categorical = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])
    # sklearn version-agnostic OneHotEncoder kw
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", ohe),
    ])
    pre = ColumnTransformer([("num", num_tf, numeric), ("cat", cat_tf, categorical)])
    model = GradientBoostingClassifier(random_state=0)
    return Pipeline([("pre", pre), ("clf", model)])


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    test_ids = test["PassengerId"].copy()

    y = train["Transported"].astype(bool).astype(int)
    X = feature_engineer(train.drop(columns=["Transported"]))
    X_test = feature_engineer(test)

    pipe = build_pipeline()

    # CV accuracy as headline estimate
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=1)
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

    pipe.fit(X, y)
    train_acc = float((pipe.predict(X) == y).mean())
    preds = pipe.predict(X_test).astype(bool)

    sub = pd.DataFrame({"PassengerId": test_ids, "Transported": preds})
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"Wrote submission.csv with {len(sub)} rows")

    result = {
        "target_column_chosen": "Transported",
        "headline_metric": "accuracy_cv5_mean",
        "headline_value": cv_mean,
        "all_metrics": {
            "accuracy_cv5_mean": cv_mean,
            "accuracy_cv5_std": cv_std,
            "accuracy_cv5_folds": [float(s) for s in cv_scores],
            "train_accuracy": train_acc,
        },
        "interpretation_notes": (
            "Spec says 'predict the missing target column'; only Transported is "
            "absent from test.csv, so it is the label. Metric is classification "
            "accuracy per the spec. Headline value is the mean of stratified-free "
            "5-fold CV accuracy on train (no held-out test labels available)."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote result.json")


if __name__ == "__main__":
    main()
