"""
Titanic — predict the column missing from test.csv.

Target-column reasoning (structural, not by name-matching):
  - train.csv has columns: PassengerId, Survived, Pclass, Name, Sex, Age,
        SibSp, Parch, Ticket, Fare, Cabin, Embarked
  - test.csv has every train column EXCEPT `Survived`.
  - The spec explicitly says: "the target prediction column is missing — you
        must predict it." The one and only column structurally missing from
        test is `Survived`. So target = Survived.
  - `gender_submission.csv` is a RED HERRING for row alignment: its
        PassengerIds are 892..1309 (the classic Kaggle public test set),
        but our actual test.csv has PassengerIds 6..890 (a re-split of the
        original Kaggle training set). We must align our submission to the
        ACTUAL test.csv PassengerIds, not gender_submission.

Metric: classification accuracy (per the spec).

Model: Logistic regression and Random Forest, picked by 5-fold stratified
CV accuracy. We also report majority-class baseline and the
"all-females-survive" heuristic baseline for context.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------- paths ----------
HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v1_dsb_target_ambig/titanic/data")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

SUBMISSION_PATH = HERE / "submission.csv"
RESULT_PATH = HERE / "result.json"

RANDOM_STATE = 42


# ---------- feature engineering ----------
TITLE_RE = re.compile(r",\s*([^.]+)\.")


def extract_title(name: str) -> str:
    m = TITLE_RE.search(str(name))
    if not m:
        return "None"
    title = m.group(1).strip()
    # collapse rare titles
    mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
        "the Countess": "Rare",
    }
    return mapping.get(title, title if title in {"Mr", "Mrs", "Miss", "Master"} else "Rare")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Pclass"] = df["Pclass"].astype(int)
    out["Sex"] = df["Sex"].astype(str)
    out["Age"] = df["Age"].astype(float)
    out["SibSp"] = df["SibSp"].astype(int)
    out["Parch"] = df["Parch"].astype(int)
    out["Fare"] = df["Fare"].astype(float)
    out["Embarked"] = df["Embarked"].fillna("S").astype(str)
    out["Title"] = df["Name"].map(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = df["Cabin"].notna().astype(int)
    return out


NUMERIC_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin", "Pclass"]
CATEGORICAL_COLS = ["Sex", "Embarked", "Title"]


def make_preprocessor() -> ColumnTransformer:
    num = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    cat = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", num, NUMERIC_COLS), ("cat", cat, CATEGORICAL_COLS)])


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # ----- structural sanity check: confirm the target -----
    target_col_candidates = sorted(set(train.columns) - set(test.columns))
    assert target_col_candidates == ["Survived"], (
        f"Expected exactly one missing column ('Survived') in test, "
        f"got: {target_col_candidates}"
    )
    target = "Survived"

    y = train[target].astype(int).values
    X_train = build_features(train)
    X_test = build_features(test)

    # ----- baselines for context -----
    majority_baseline_acc = float(max((y == 0).mean(), (y == 1).mean()))
    # "all-females-survive" heuristic on TRAIN (sanity check; gender_submission
    # in the data dir encodes this rule, but for different PassengerIds)
    gender_rule_pred_train = (train["Sex"].values == "female").astype(int)
    gender_rule_acc_train = float(accuracy_score(y, gender_rule_pred_train))

    # ----- CV model selection -----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pre = make_preprocessor()

    lr_pipe = Pipeline(
        [("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE))]
    )
    rf_pipe = Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    lr_scores = cross_val_score(lr_pipe, X_train, y, cv=cv, scoring="accuracy", n_jobs=-1)
    rf_scores = cross_val_score(rf_pipe, X_train, y, cv=cv, scoring="accuracy", n_jobs=-1)

    lr_mean, rf_mean = float(lr_scores.mean()), float(rf_scores.mean())

    if rf_mean >= lr_mean:
        best_name, best_pipe, best_mean, best_std = (
            "random_forest",
            rf_pipe,
            rf_mean,
            float(rf_scores.std()),
        )
    else:
        best_name, best_pipe, best_mean, best_std = (
            "logistic_regression",
            lr_pipe,
            lr_mean,
            float(lr_scores.std()),
        )

    # Fit best on full train, predict test
    best_pipe.fit(X_train, y)
    preds = best_pipe.predict(X_test).astype(int)

    # ----- write submission, aligned to test.csv PassengerIds (NOT gender_submission!) -----
    submission = pd.DataFrame({"PassengerId": test["PassengerId"].values, target: preds})
    # one row per test row, original order preserved
    assert len(submission) == len(test), "submission row count must match test"
    submission.to_csv(SUBMISSION_PATH, index=False)

    # ----- result.json -----
    result = {
        "target_column_chosen": target,
        "target_column_justification": (
            "Structural check: train.csv has the columns "
            f"{sorted(train.columns)} and test.csv has {sorted(test.columns)}; "
            "the only column in train but absent from test is 'Survived'. "
            "The spec also explicitly defines Survived (1=survived, 0=not) "
            "and asks the user to predict the missing target column. "
            "Alternative candidates (e.g., Sex) are present in test.csv and "
            "are therefore NOT held out, so cannot be the target."
        ),
        "headline_metric": "accuracy",
        "headline_value": round(best_mean, 6),
        "all_metrics": {
            "cv_accuracy_logistic_regression_mean": round(lr_mean, 6),
            "cv_accuracy_logistic_regression_std": round(float(lr_scores.std()), 6),
            "cv_accuracy_random_forest_mean": round(rf_mean, 6),
            "cv_accuracy_random_forest_std": round(float(rf_scores.std()), 6),
            "selected_model": best_name,
            "selected_model_cv_accuracy_mean": round(best_mean, 6),
            "selected_model_cv_accuracy_std": round(best_std, 6),
            "baseline_majority_class_train_accuracy": round(majority_baseline_acc, 6),
            "baseline_all_females_survive_train_accuracy": round(gender_rule_acc_train, 6),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_passengerid_range": [int(train["PassengerId"].min()), int(train["PassengerId"].max())],
            "test_passengerid_range": [int(test["PassengerId"].min()), int(test["PassengerId"].max())],
        },
        "anticipated_probes": [
            "Target-column choice on structural grounds: confirmed by set-diff of "
            "train vs test columns; only 'Survived' is missing. We assert this in code.",
            "Submission row alignment: rows ordered by test.csv's own PassengerId "
            "(in original test.csv order), not gender_submission.csv. "
            "gender_submission.csv covers PassengerIds 892-1309 (the classic Kaggle "
            "public leaderboard test set) but THIS test.csv has PassengerIds 6-890, "
            "i.e. a re-split of the original 891-row training set. We use test.csv.",
            "Choice of metric: classification accuracy, exactly as the spec states.",
            "Baseline framing: we report majority-class (0.623) and the "
            "all-females-survive heuristic (0.787 on train) so the headline model "
            "score has context.",
            "Data quirks: Age has ~20% missing, Cabin >75% missing, Embarked has 2 "
            "missing in train. Handled with median impute for numerics, "
            "most-frequent for categoricals, and a HasCabin indicator instead of "
            "trying to use the noisy Cabin string. Title is extracted from Name to "
            "recover signal lost when imputing Age.",
            "Row count: spec says 713/180 but files are 712/179 (off by one each). "
            "We use the actual file lengths; one row per test row, header included.",
            "Headline_value is the CV estimate (out-of-fold mean accuracy), not "
            "train accuracy, so it is not optimistically biased by overfitting.",
        ],
        "interpretation_notes": (
            "Only Survived is structurally missing from test.csv, so that is the "
            "target. The provided gender_submission.csv is for a different "
            "PassengerId range and would mis-align rows if copied verbatim; we "
            "ignore its rows and use only its column-name convention "
            "(PassengerId, Survived). Models compared by 5-fold stratified CV "
            "accuracy; the better of logistic regression and random forest is "
            "selected and refit on the full training set before predicting test."
        ),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # Console summary
    print(f"target_column_chosen: {target}")
    print(f"LR CV accuracy: {lr_mean:.4f} ± {lr_scores.std():.4f}")
    print(f"RF CV accuracy: {rf_mean:.4f} ± {rf_scores.std():.4f}")
    print(f"Selected: {best_name} (acc={best_mean:.4f})")
    print(f"Baselines: majority={majority_baseline_acc:.4f}, "
          f"females-survive(train)={gender_rule_acc_train:.4f}")
    print(f"Submission rows: {len(submission)} (test rows: {len(test)})")
    print(f"Wrote {SUBMISSION_PATH}")
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
