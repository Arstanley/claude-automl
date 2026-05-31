"""Titanic survival prediction.

Skills applied (mechanically from skill library INDEX.md):
  - skill_missing_indicator_when_missingness_signals_class: Age (~20%) and Cabin (~78%) NaN
    -> add *_missing flags; NA-as-category for Embarked/Cabin
  - skill_stringify_low_cardinality_int_codes: Pclass is a 3-level int code -> treat as categorical
  - skill_tune_when_n_train_supports_it: n_train=712 >= 500, no size/latency/production tokens
    -> small explicit grid over GradientBoosting params

Skills evaluated and NOT applied:
  - skill_probe_group_structure_in_ids: PassengerId disjoint; Ticket overlap small (46) and no
    clean group structure governing survival; family-name parsing already captures relevant signal
  - skill_train_on_listed_auxiliary_files: prompt lists no auxiliary data files
  - skill_verify_split_claim_before_holdout: prompt makes no structural train/test split claim
  - skill_multi_axis_constraint_budgets: no latency/size/hard-rule constraints in prompt
  - skill_tune_decision_threshold_for_f1: metric is accuracy (threshold ~argmax of P, 0.5 is
    Bayes-optimal under accuracy with calibrated probs); skipped
  - skill_log_target_for_skewed_nonnegative_regression: this is classification
  - skill_blend_two_model_families: n_train=712 < 1000 precondition gate fails
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/titanic/data")

t0 = time.time()


# ----------------------------- feature engineering -----------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing indicators (skill_missing_indicator_when_missingness_signals_class)
    df["Age_missing"] = df["Age"].isna().astype(int)
    df["Cabin_missing"] = df["Cabin"].isna().astype(int)

    # Pclass: stringify low-cardinality int code (skill_stringify_low_cardinality_int_codes)
    df["Pclass_s"] = df["Pclass"].astype(str)

    # Title from Name
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).fillna("Unk")
    # Collapse rare titles
    common = {"Mr", "Mrs", "Miss", "Master"}
    df["Title"] = df["Title"].where(df["Title"].isin(common), other="Rare")

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Cabin deck (first letter); 'U' for unknown -> NA-as-category
    df["Deck"] = df["Cabin"].str[0].fillna("U")

    # Ticket prefix (non-numeric token); strips spaces/dots
    def tprefix(t: str) -> str:
        t = str(t)
        m = re.match(r"([A-Za-z./]+)", t)
        return m.group(1).replace(".", "").replace("/", "").upper() if m else "NUM"

    df["TicketPrefix"] = df["Ticket"].apply(tprefix)
    # Collapse rare prefixes
    vc = df["TicketPrefix"].value_counts()
    rare = set(vc[vc < 5].index)
    df["TicketPrefix"] = df["TicketPrefix"].where(~df["TicketPrefix"].isin(rare), other="RARE")

    # Embarked NA -> 'U' (NA as category)
    df["Embarked"] = df["Embarked"].fillna("U")

    return df


# ----------------------------- load data -----------------------------

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

train_fe = add_features(train)
test_fe = add_features(test)

TARGET = "Survived"
ID_COL = "PassengerId"

num_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "Age_missing", "Cabin_missing"]
cat_features = ["Sex", "Pclass_s", "Title", "Deck", "Embarked", "TicketPrefix"]

X = train_fe[num_features + cat_features]
y = train_fe[TARGET].values
X_test = test_fe[num_features + cat_features]

# ----------------------------- preprocessing -----------------------------

# OneHotEncoder version-compat (sparse_output >= 1.2)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", ohe, cat_features),
    ]
)

# ----------------------------- model search (small grid) -----------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = [
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
]

best = None
results = []
for cfg in grid:
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", GradientBoostingClassifier(random_state=42, **cfg)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=2)
    mean = float(scores.mean())
    std = float(scores.std())
    results.append({"cfg": cfg, "cv_acc_mean": mean, "cv_acc_std": std})
    if best is None or mean > best["cv_acc_mean"]:
        best = {"cfg": cfg, "cv_acc_mean": mean, "cv_acc_std": std}

# Also baseline RandomForest for sanity (not blended; n<1000 gates blend skill off)
rf_pipe = Pipeline(
    steps=[
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2,
                                       random_state=42, n_jobs=2)),
    ]
)
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="accuracy", n_jobs=2)
rf_mean = float(rf_scores.mean())
results.append({"cfg": {"model": "RandomForest", "n_estimators": 500}, "cv_acc_mean": rf_mean,
                "cv_acc_std": float(rf_scores.std())})

# Pick best overall (still single-model; blend skipped per precondition)
if rf_mean > best["cv_acc_mean"]:
    final_pipe = rf_pipe
    final_label = "RandomForest(n=500)"
    final_mean = rf_mean
else:
    final_pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", GradientBoostingClassifier(random_state=42, **best["cfg"])),
        ]
    )
    final_label = f"GBM({best['cfg']})"
    final_mean = best["cv_acc_mean"]

# ----------------------------- fit + predict -----------------------------

final_pipe.fit(X, y)
preds = final_pipe.predict(X_test).astype(int)

submission = pd.DataFrame({ID_COL: test_fe[ID_COL].values, TARGET: preds})
submission.to_csv(HERE / "submission.csv", index=False)

elapsed = time.time() - t0

# ----------------------------- result.json -----------------------------

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "accuracy",
    "headline_value": final_mean,
    "all_metrics": {
        "cv_accuracy_mean": final_mean,
        "cv_grid_results": results,
        "final_model": final_label,
    },
    "skills_applied": [
        "skill_missing_indicator_when_missingness_signals_class.md",
        "skill_stringify_low_cardinality_int_codes.md",
        "skill_tune_when_n_train_supports_it.md",
    ],
    "skills_evaluated_not_applied": [
        {"skill": "skill_probe_group_structure_in_ids.md",
         "reason": "PassengerId disjoint train/test; Ticket overlap only 46 of 712 train tickets; no clean group structure that governs survival beyond family/title features already added."},
        {"skill": "skill_train_on_listed_auxiliary_files.md",
         "reason": "Prompt lists no auxiliary data files; only train.csv, test.csv, gender_submission.csv (example only)."},
        {"skill": "skill_verify_split_claim_before_holdout.md",
         "reason": "Prompt makes no structural claim about how train/test were split (no temporal or group claim)."},
        {"skill": "skill_multi_axis_constraint_budgets.md",
         "reason": "Prompt names no latency/size/hard-rule constraints; only metric is accuracy."},
        {"skill": "skill_tune_decision_threshold_for_f1.md",
         "reason": "Metric is accuracy, not F1/F-beta/MCC/balanced-accuracy; 0.5 (argmax) is the Bayes-optimal threshold under accuracy for calibrated probs."},
        {"skill": "skill_log_target_for_skewed_nonnegative_regression.md",
         "reason": "This is binary classification, not regression."},
        {"skill": "skill_blend_two_model_families.md",
         "reason": "Precondition n_train >= 1000 fails (n_train=712); RF was scored alongside GBM but only as a single-model alternative, not blended."},
    ],
    "interpretation_notes": (
        "Titanic with n=712: engineered Title/Deck/FamilySize/TicketPrefix features, added "
        "Age_missing and Cabin_missing indicators, stringified Pclass, and ran a small 5-cfg "
        "GradientBoosting grid plus RF baseline under 5-fold StratifiedKFold. Selected the "
        f"best CV configuration ({final_label}, CV acc={final_mean:.4f}). Single model "
        "shipped; blend skill gated off by n<1000 precondition."
    ),
    "wall_clock_seconds": round(elapsed, 2),
}

with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Wrote submission.csv ({len(submission)} rows) and result.json")
print(f"Final model: {final_label}  CV accuracy: {final_mean:.4f}  elapsed={elapsed:.1f}s")
