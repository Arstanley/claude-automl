"""
Solution for playground-series-s3e22 (Horse Survival, micro-F1).

Skills applied (from harness skill library):
- skill_stringify_low_cardinality_int_codes  (lesion_1: 56 unique 4-digit codes)
- skill_missing_indicator_when_missingness_signals_class  (abdomen ~18%, rectal_exam_feces ~15% missing)
- skill_tune_when_n_train_supports_it  (n_train=988 >= 500; no constraint tokens in prompt)
- skill_probe_group_structure_in_ids  (hospital_number overlaps 90% train/test; logged + dropped)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e22/data/")
PROMPT_PATH = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e22/full_prompt.txt")

TARGET = "outcome"
ID_COL = "id"
RANDOM_STATE = 0


# ------------------------------------------------------------------ data load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
prompt = PROMPT_PATH.read_text()

print(f"train: {train.shape}  test: {test.shape}")
print(f"target dist: {train[TARGET].value_counts().to_dict()}")


# ----------------------------------------- skill: probe_group_structure_in_ids
# hospital_number has 90% train/test overlap. It is a hospital id, not a row
# group — multiple horses can come from the same hospital, but using it as a
# feature would just memorize hospital→outcome. Drop it. Also too few rows
# per hospital to make GroupKFold meaningful (988 rows / many hospitals).
group_probe = {}
for c in train.columns:
    if c == TARGET or c not in test.columns:
        continue
    a, b = set(train[c].astype(str)), set(test[c].astype(str))
    if not b:
        continue
    leakage = len(a & b) / len(b)
    if leakage >= 0.05 and c not in (ID_COL,):
        group_probe[c] = round(leakage, 4)
print(f"group-leakage probe: {group_probe}")

# Drop id and hospital_number
DROP_COLS = [ID_COL, "hospital_number"]


# ------------------------- skill: stringify_low_cardinality_int_codes (lesion_1 etc.)
stringified = []
for c in train.select_dtypes(include="integer").columns:
    if c == TARGET or c.lower() in {"id", "index", "rowid"} or c in DROP_COLS:
        continue
    nuniq = train[c].nunique()
    if 1 < nuniq <= max(100, int(0.05 * len(train))):
        if train[c].is_monotonic_increasing and nuniq == len(train):
            continue
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        stringified.append(c)
print(f"stringified code columns: {stringified}")


# ---------------- skill: missing_indicator_when_missingness_signals_class
indicators_added = []
for c in train.columns:
    if c == TARGET or c in DROP_COLS:
        continue
    miss_rate_tr = train[c].isna().mean()
    if miss_rate_tr >= 0.05:
        ind = f"{c}_missing"
        train[ind] = train[c].isna().astype(int)
        if c in test.columns:
            test[ind] = test[c].isna().astype(int)
        indicators_added.append(c)
# NA-as-category for the high-missingness categoricals
for c in indicators_added:
    if train[c].dtype == object:
        train[c] = train[c].fillna("__missing__")
        if c in test.columns:
            test[c] = test[c].fillna("__missing__")
print(f"missing-indicators added for: {indicators_added}")


# -------------------------------------------------------- feature matrices
y = train[TARGET].values
X_train = train.drop(columns=[TARGET] + DROP_COLS)
X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

# Align test column order to train
X_test = X_test[X_train.columns]

cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
print(f"n_cat={len(cat_cols)}  n_num={len(num_cols)}")


# ----------------------------------------------------------------- pipeline
# HistGradientBoosting natively handles NaN in numeric columns, but we still
# impute the categorical NaN as a literal (done via NA-as-category above for
# flagged cols, and via OneHotEncoder for the rest with the implicit "nan" str).
def make_pipe(**hgb_kwargs):
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", "passthrough", num_cols),
    ])
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **hgb_kwargs)
    return Pipeline([("pre", pre), ("model", model)])


# ------------------ skill: tune_when_n_train_supports_it (n=988 >= 500; no constraint tokens)
constraint_tokens = re.findall(r"\b(size|latency|MB|ms|production|mobile|p99)\b", prompt, re.IGNORECASE)
should_tune = len(train) >= 500 and not constraint_tokens
print(f"should_tune={should_tune} (n_train={len(train)}, constraint_tokens={constraint_tokens})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = "f1_micro"  # task metric

if should_tune:
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=1.0),
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=15, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=1.0),
        dict(max_iter=400, learning_rate=0.04, max_depth=6,    max_leaf_nodes=31, l2_regularization=0.5),
    ]
    cv_results = []
    for params in grid:
        scores = cross_val_score(make_pipe(**params), X_train, y, cv=cv, scoring=scoring, n_jobs=1)
        cv_results.append({"params": params, "mean": float(scores.mean()), "std": float(scores.std())})
        print(f"  {params}: {scores.mean():.4f} +/- {scores.std():.4f}")
    cv_results.sort(key=lambda r: r["mean"], reverse=True)
    best_params = cv_results[0]["params"]
    best_cv = cv_results[0]["mean"]
    best_cv_std = cv_results[0]["std"]
else:
    best_params = dict(max_iter=200, learning_rate=0.05, max_depth=None, max_leaf_nodes=31)
    scores = cross_val_score(make_pipe(**best_params), X_train, y, cv=cv, scoring=scoring, n_jobs=1)
    cv_results = [{"params": best_params, "mean": float(scores.mean()), "std": float(scores.std())}]
    best_cv = float(scores.mean())
    best_cv_std = float(scores.std())

print(f"best CV micro-F1: {best_cv:.4f} +/- {best_cv_std:.4f}  params={best_params}")


# --------------------------------------------------------------- final fit
final = make_pipe(**best_params).fit(X_train, y)
preds = final.predict(X_test)

# Build submission
sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: preds})
sub.to_csv(HERE / "submission.csv", index=False)
print(f"submission written: {HERE / 'submission.csv'}  rows={len(sub)}")

# Sanity: per-fold predicted-class distribution
print("predicted dist:", pd.Series(preds).value_counts().to_dict())


# ------------------------------------------------------------------- result.json
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "micro_f1_5fold_stratified_cv",
    "headline_value": best_cv,
    "all_metrics": {
        "cv_micro_f1_mean": best_cv,
        "cv_micro_f1_std": best_cv_std,
        "cv_n_splits": 5,
    },
    "skills_applied": [
        "skill_stringify_low_cardinality_int_codes.md",
        "skill_missing_indicator_when_missingness_signals_class.md",
        "skill_tune_when_n_train_supports_it.md",
        "skill_probe_group_structure_in_ids.md",
    ],
    "interpretation_notes": (
        "3-class horse-survival classification (n_train=988, micro-F1). HistGradientBoosting on "
        "OHE-categorical + passthrough-numeric. Stringified lesion_1/2/3 (low-card int codes), added "
        "missing-indicators + NA-as-category on the >=5% missing columns (abdomen, rectal_exam_feces, "
        "nasogastric_tube, peripheral_pulse), dropped hospital_number (90% train/test overlap = id leak), "
        "and selected hyperparams from a 5-config grid by stratified 5-fold CV."
    ),
    "audit": {
        "stringified_columns": stringified,
        "missingness_indicators_added": indicators_added,
        "group_leakage_probe": group_probe,
        "dropped_columns": DROP_COLS,
        "hyperparameter_grid": cv_results,
        "chosen_config": best_params,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    },
}
with open(HERE / "result.json", "w") as fh:
    json.dump(result, fh, indent=2, default=str)
print(f"result.json written: {HERE / 'result.json'}")
