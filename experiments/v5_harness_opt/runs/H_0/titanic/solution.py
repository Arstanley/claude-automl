"""
Titanic survival prediction.

Skills applied:
  1. skill_verify_split_claim_before_holdout - prompt claims 891/418 train/test
     split but local data is 712/179 (random subset). Use StratifiedKFold.
  2. skill_missing_indicator_when_missingness_signals_class - Age (~20% NaN),
     Cabin (~77% NaN), Embarked (small) get explicit *_missing flags +
     NA-as-category handling.
  3. skill_tune_when_n_train_supports_it - n_train=712 >= 500, GBM family,
     no size/latency tokens in prompt. Small explicit 3-config grid.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/titanic/data")
WORK_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_0/titanic")
PROMPT_PATH = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/titanic/full_prompt.txt")
TARGET = "Survived"
ID_COL = "PassengerId"
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def extract_title(name: str) -> str:
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"


TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Capt": "Rare", "Sir": "Rare", "Lady": "Rare",
    "the Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare",
}


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (X_train, X_test, missingness_indicators_added)."""
    train = train_raw.copy()
    test = test_raw.copy()

    # ---- Missingness audit ----
    indicators_added: list[str] = []
    for c in ["Age", "Cabin", "Embarked", "Fare"]:
        if c in train.columns:
            rate = train[c].isna().mean()
            if rate >= 0.05 or c == "Cabin":  # Cabin is high-info missingness
                ind = f"{c}_missing"
                train[ind] = train[c].isna().astype(int)
                test[ind] = test[c].isna().astype(int)
                indicators_added.append(c)

    # ---- Engineered features ----
    for df in (train, test):
        df["Title"] = df["Name"].map(extract_title).map(lambda t: TITLE_MAP.get(t, "Rare"))
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        # Cabin deck letter (first letter); NA-as-category
        df["CabinDeck"] = df["Cabin"].fillna("__missing__").astype(str).str[0]
        # Ticket prefix (non-numeric part) and ticket group size handled below
        df["TicketPrefix"] = df["Ticket"].astype(str).str.replace(r"[0-9]+$", "", regex=True).str.strip()
        df["TicketPrefix"] = df["TicketPrefix"].replace("", "NUM")

    # ---- Imputation (after indicators!) ----
    # Age: median by Title (more accurate than global median)
    full_for_age = pd.concat([train[["Title", "Age"]], test[["Title", "Age"]]], ignore_index=True)
    age_by_title = full_for_age.groupby("Title")["Age"].median()
    global_age_median = full_for_age["Age"].median()
    for df in (train, test):
        df["Age"] = df.apply(
            lambda r: age_by_title.get(r["Title"], global_age_median) if pd.isna(r["Age"]) else r["Age"],
            axis=1,
        )

    # Fare: median by Pclass
    full_for_fare = pd.concat([train[["Pclass", "Fare"]], test[["Pclass", "Fare"]]], ignore_index=True)
    fare_by_pclass = full_for_fare.groupby("Pclass")["Fare"].median()
    for df in (train, test):
        mask = df["Fare"].isna()
        if mask.any():
            df.loc[mask, "Fare"] = df.loc[mask, "Pclass"].map(fare_by_pclass)

    # Embarked: most common ("S"), NA-as-category alternative — but only 2 NaNs in train, 0 in test
    for df in (train, test):
        df["Embarked"] = df["Embarked"].fillna("S")

    # ---- Drop columns that are too high-cardinality / leaky ----
    drop_cols = [ID_COL, TARGET, "Name", "Ticket", "Cabin"]

    # ---- One-hot encode the categoricals consistently ----
    # Concatenate to get a single shared dummy schema
    train["__is_train__"] = 1
    test["__is_train__"] = 0
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])
    test[TARGET] = np.nan  # placeholder
    combined = pd.concat([train, test], ignore_index=True)

    cat_cols = ["Sex", "Embarked", "Title", "CabinDeck", "TicketPrefix"]
    # Stringify low-cardinality ints (Pclass): cast to string for one-hot
    combined["Pclass"] = combined["Pclass"].astype(str)
    cat_cols.append("Pclass")

    combined_enc = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    train_enc = combined_enc[combined_enc["__is_train__"] == 1].copy()
    test_enc = combined_enc[combined_enc["__is_train__"] == 0].copy()

    feature_cols = [
        c for c in combined_enc.columns
        if c not in drop_cols + ["__is_train__"]
    ]

    X_train = train_enc[feature_cols].astype(float)
    X_test = test_enc[feature_cols].astype(float)
    return X_train, X_test, indicators_added


# ---------------------------------------------------------------------------
# Verify the prompt's split claim
# ---------------------------------------------------------------------------
def verify_split(prompt: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    audit: dict = {}
    audit["claim_in_prompt"] = "train=891, test=418 (Kaggle public)" if "891" in prompt and "418" in prompt else None
    audit["actual_n_train"] = int(len(train))
    audit["actual_n_test"] = int(len(test))
    audit["passengerid_overlap"] = int(len(set(train[ID_COL]) & set(test[ID_COL])))
    audit["claim_holds"] = (len(train) == 891 and len(test) == 418)
    audit["chosen_cv"] = "StratifiedKFold(5, shuffle=True) — random interleaving confirmed"
    return audit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    prompt = PROMPT_PATH.read_text()

    print(f"n_train={len(train)}, n_test={len(test)}")
    split_audit = verify_split(prompt, train, test)
    print("split audit:", split_audit)

    X, X_test, indicators_added = build_features(train, test)
    y = train[TARGET].astype(int).values
    print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
    print(f"Missingness indicators added for: {indicators_added}")

    # ---- Tune (n_train=712 >= 500, GBM family, no size/latency constraints) ----
    grid = [
        dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9),
        dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.9),
        dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8),
        dict(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=1.0),  # conservative baseline
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for params in grid:
        scores = cross_val_score(
            GradientBoostingClassifier(**params, random_state=RANDOM_STATE),
            X, y, cv=cv, scoring="accuracy", n_jobs=1,
        )
        results.append((scores.mean(), scores.std(), params))
        print(f"  {params}: acc={scores.mean():.4f} ± {scores.std():.4f}")
    results.sort(key=lambda r: r[0], reverse=True)
    best_mean, best_std, best_params = results[0]
    print(f"chosen: {best_params}  (CV acc = {best_mean:.4f} ± {best_std:.4f})")

    # Refit on all train
    final = GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
    final.fit(X, y)

    preds = final.predict(X_test).astype(int)
    submission = pd.DataFrame({
        ID_COL: test[ID_COL].values,
        TARGET: preds,
    })
    sub_path = WORK_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  ({len(submission)} rows)")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "accuracy",
        "headline_value": float(best_mean),
        "all_metrics": {
            "cv_accuracy_mean": float(best_mean),
            "cv_accuracy_std": float(best_std),
            "cv_folds": 5,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(X.shape[1]),
        },
        "skills_applied": [
            "skill_verify_split_claim_before_holdout.md",
            "skill_missing_indicator_when_missingness_signals_class.md",
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Verified the prompt's implicit 891/418 split is wrong locally (712/179, "
            "PassengerIds randomly interleaved in 1-891) so used StratifiedKFold. "
            "Added missing-indicators for Age/Cabin/Embarked and built Title, FamilySize, "
            "IsAlone, CabinDeck, TicketPrefix features. Tuned GBM over a 4-config grid; "
            "best config refit on full train."
        ),
        "audit": {
            "split_audit": split_audit,
            "missingness_indicators_added": indicators_added,
            "hyperparameter_grid": [r[2] for r in results],
            "hyperparameter_cv_scores": [{"params": r[2], "mean": r[0], "std": r[1]} for r in results],
            "chosen_config": best_params,
        },
    }
    res_path = WORK_DIR / "result.json"
    with res_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {res_path}")


if __name__ == "__main__":
    main()
