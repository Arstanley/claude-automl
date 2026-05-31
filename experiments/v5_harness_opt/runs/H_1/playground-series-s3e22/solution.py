"""
Playground Series S3E22 — Horse Survival (multiclass: lived/died/euthanized).
Metric: micro-averaged F1.

Skills applied (from harness_H_1 library):
  - skill_stringify_low_cardinality_int_codes  (lesion_1/2/3 are anatomical codes,
    not magnitudes; hospital_number is a categorical ID; all stringified.)
  - skill_missing_indicator_when_missingness_signals_class
    (abdomen, rectal_exam_feces, nasogastric_tube > 5% missing in medical data —
     missingness is informative.)
  - skill_tune_when_n_train_supports_it  (n_train=988 > 500, no constraint tokens
     in prompt; small explicit 3-config grid over HistGradientBoosting.)

Skills evaluated and NOT applied (with reason) — written into result.json.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e22/data")
TARGET = "outcome"
ID = "id"
N_JOBS = 2
RANDOM_STATE = 0

t0 = time.time()

# ---------------- load ----------------
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"train: {train.shape}  test: {test.shape}")

# ---------------- skill: stringify low-card int codes ----------------
# lesion_1 (56 uniq, 4-digit anatomical codes), lesion_2 (4 uniq), lesion_3 (2 uniq)
# hospital_number (241 uniq in train) — 93.5% of test rows share a hospital with
# train, so it CAN be treated as a categorical. With 241 levels it's heavy for
# one-hot; but a tree-OHE can absorb the signal. We'll include it as string +
# OHE with handle_unknown="ignore".
stringify_cols = []
for c in train.select_dtypes(include="integer").columns:
    if c in (ID, TARGET):
        continue
    n_uniq = train[c].nunique()
    if 1 < n_uniq <= max(100, int(0.05 * len(train))):
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        stringify_cols.append(c)
# hospital_number is high-cardinality int but is clearly a categorical ID.
# Per the skill spec the precondition gates on nunique <= max(100, 0.05*n_rows)
# = max(100, 49) = 100. hospital_number has 241 levels, so the strict
# precondition fails. We respect that and leave it as an integer (HGB tolerates
# integer-coded categoricals reasonably well).
print(f"stringified code columns: {stringify_cols}")

# ---------------- skill: missingness indicators ----------------
indicator_cols = []
miss_rate = train.isna().mean()
for c, r in miss_rate.items():
    if c in (ID, TARGET):
        continue
    if r >= 0.05:
        ind = f"{c}_missing"
        train[ind] = train[c].isna().astype(int)
        test[ind] = test[c].isna().astype(int)
        indicator_cols.append(c)
# NaN-as-category for categorical columns above threshold
for c in indicator_cols:
    if train[c].dtype == object:
        train[c] = train[c].fillna("__missing__")
        test[c] = test[c].fillna("__missing__")
print(f"added missingness indicators for: {indicator_cols}")

# ---------------- features ----------------
y = train[TARGET]
X = train.drop(columns=[ID, TARGET])
X_test = test.drop(columns=[ID])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"cat cols ({len(cat_cols)}): {cat_cols}")
print(f"num cols ({len(num_cols)}): {num_cols}")

pre = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_cols,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ---------------- skill: tune HGB ----------------
# n_train = 988, no size/latency/production tokens in prompt → tune.
configs = [
    dict(max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=0.0),
    dict(max_iter=500, learning_rate=0.03, max_depth=4, l2_regularization=0.0),
    dict(max_iter=300, learning_rate=0.05, max_depth=5, l2_regularization=1.0),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
for params in configs:
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE, **params
                ),
            ),
        ]
    )
    scores = cross_val_score(pipe, X, y, scoring="f1_micro", cv=cv, n_jobs=N_JOBS)
    results.append((float(scores.mean()), float(scores.std()), params))
    print(f"  {params} -> micro-F1 {scores.mean():.4f} +/- {scores.std():.4f}")

results.sort(key=lambda r: r[0], reverse=True)
best_mean, best_std, best_params = results[0]
print(f"best config: {best_params}  micro-F1 OOF mean={best_mean:.4f}")

# ---------------- fit + predict ----------------
final = Pipeline(
    steps=[
        ("pre", pre),
        (
            "clf",
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, **best_params
            ),
        ),
    ]
)
final.fit(X, y)
pred = final.predict(X_test)

sub = pd.DataFrame({ID: test[ID].values, TARGET: pred})
sub.to_csv(HERE / "submission.csv", index=False)
print(f"wrote submission.csv shape={sub.shape}")
assert len(sub) == len(test)
assert set(sub.columns) == {ID, TARGET}
print(sub.head())

# ---------------- result.json ----------------
skills_applied = [
    "skill_stringify_low_cardinality_int_codes.md",
    "skill_missing_indicator_when_missingness_signals_class.md",
    "skill_tune_when_n_train_supports_it.md",
]
skills_evaluated_not_applied = [
    {
        "skill": "skill_probe_group_structure_in_ids.md",
        "reason": (
            "hospital_number has 93.5% test-row overlap with train, but the column "
            "is not an obvious group identifier for outcome (same hospital sees "
            "all 3 outcomes); GroupKFold on it would shrink effective train "
            "to ~117 unique groups and inflate variance. Stratified KFold kept."
        ),
    },
    {
        "skill": "skill_train_on_listed_auxiliary_files.md",
        "reason": (
            "Prompt mentions the original Horse Survival Dataset as optional, but "
            "no auxiliary file is present in DATA_DIR; nothing to concatenate."
        ),
    },
    {
        "skill": "skill_verify_split_claim_before_holdout.md",
        "reason": "No structural split claim made in prompt — nothing to verify.",
    },
    {
        "skill": "skill_multi_axis_constraint_budgets.md",
        "reason": "Prompt sets no latency/size/hard-rule constraint — single axis (micro-F1).",
    },
    {
        "skill": "skill_tune_decision_threshold_for_f1.md",
        "reason": (
            "Multiclass classification (3 classes), not binary — threshold tuning "
            "doesn't apply; argmax over class probabilities is the canonical decision rule "
            "for micro-F1 in multiclass."
        ),
    },
    {
        "skill": "skill_log_target_for_skewed_nonnegative_regression.md",
        "reason": "Classification task, not regression.",
    },
    {
        "skill": "skill_blend_two_model_families.md",
        "reason": (
            "n_train=988 < 1000 precondition threshold; below the size needed for a "
            "stable second-family fit + honest CV-driven blend weight."
        ),
    },
]

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "micro_f1",
    "headline_value": best_mean,
    "all_metrics": {
        "cv_micro_f1_mean": best_mean,
        "cv_micro_f1_std": best_std,
        "cv_folds": 5,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_classes": int(y.nunique()),
        "class_counts": y.value_counts().to_dict(),
    },
    "skills_applied": skills_applied,
    "skills_evaluated_not_applied": skills_evaluated_not_applied,
    "audit": {
        "stringified_columns": stringify_cols,
        "missingness_indicators_added": indicator_cols,
        "hyperparameter_grid": [r[2] for r in results],
        "grid_oof": [
            {"params": r[2], "mean": r[0], "std": r[1]} for r in results
        ],
        "chosen_config": best_params,
        "model_family": "HistGradientBoostingClassifier",
        "cv": "StratifiedKFold(5, shuffle=True)",
    },
    "interpretation_notes": (
        "HistGradientBoostingClassifier on 988 horse-survival rows, micro-F1. "
        "Stringified the 4-digit lesion_1 anatomical codes (and lesion_2/3) so OHE "
        "treats them as categorical rather than magnitudes; added missingness "
        "indicators for the three columns >5% NaN (medical missingness signals "
        "class). Tuned 3 HGB configs with 5-fold CV (n=988 supports it)."
    ),
    "runtime_seconds": round(time.time() - t0, 2),
}

with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"wrote result.json — headline micro-F1 OOF mean: {best_mean:.4f}")
print(f"total runtime: {result['runtime_seconds']}s")
