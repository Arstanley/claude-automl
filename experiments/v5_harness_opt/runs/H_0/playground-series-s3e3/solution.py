"""
Solution for Kaggle Playground Series S3E3: Employee Attrition prediction.
Metric: ROC AUC. Output: probability of Attrition=1 per test id.

Skills applied:
- skill_probe_group_structure_in_ids: probed id columns for leakage (none).
- skill_stringify_low_cardinality_int_codes: cast low-cardinality int codes
  (Education, JobLevel, etc.) to string for explicit categorical handling.
- skill_tune_when_n_train_supports_it: n_train=1341>=500, no constraint tokens
  in prompt -> small explicit GBM grid with StratifiedKFold-5.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e3/data")
PROMPT_PATH = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e3/full_prompt.txt")

TARGET = "Attrition"
ID_COL = "id"
RANDOM_STATE = 0


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    prompt = PROMPT_PATH.read_text()

    audit: dict = {}

    # ---- Probe 1: group leakage on id-like columns (skill_probe_group_structure_in_ids)
    leakage = {}
    for c in train.columns:
        if c == TARGET or c not in test.columns:
            continue
        a, b = set(train[c].astype(str)), set(test[c].astype(str))
        if not b:
            continue
        frac = len(a & b) / len(b)
        if frac >= 0.05:
            leakage[c] = frac
    audit["group_leakage_probe"] = leakage
    # No id column overlaps train/test -> StratifiedKFold is fine.

    # ---- Drop constant columns
    constant_cols = [c for c in train.columns if train[c].nunique() <= 1 and c != TARGET]
    audit["dropped_constant_cols"] = constant_cols
    train = train.drop(columns=constant_cols)
    test = test.drop(columns=[c for c in constant_cols if c in test.columns])

    # ---- skill_stringify_low_cardinality_int_codes
    # Tree-based final model (GBM); skill is OPTIONAL for trees but recommended
    # for code-like ints (Education, JobLevel are ordinal codes, not magnitudes;
    # PerformanceRating only takes 2 values etc.). One-hot lets the model learn
    # per-level effects without imposing a monotone ordinal.
    stringified = []
    n = len(train)
    for c in train.select_dtypes(include="integer").columns:
        if c == TARGET or c == ID_COL:
            continue
        nuniq = train[c].nunique()
        if 1 < nuniq <= max(20, int(0.02 * n)):  # tight bound: only the obvious code-like ints
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)
            stringified.append(c)
    audit["stringified_columns"] = stringified

    # ---- Build feature frame
    y = train[TARGET].astype(int).values
    X = train.drop(columns=[TARGET, ID_COL])
    X_test = test.drop(columns=[ID_COL])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    audit["cat_cols"] = cat_cols
    audit["num_cols"] = num_cols

    # OneHot for categoricals; numerics pass through (GBM doesn't need scaling).
    # handle_unknown=ignore so any test-only level is silently zero-encoded.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    # ---- skill_tune_when_n_train_supports_it
    # n_train = 1341 >= 500.
    constraint_tokens = re.findall(
        r"\b(size|latency|MB|ms|production|mobile|p99)\b", prompt, re.IGNORECASE
    )
    should_tune = len(constraint_tokens) == 0
    audit["constraint_tokens_in_prompt"] = constraint_tokens
    audit["should_tune"] = bool(should_tune)

    grid = [
        dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9),
        dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.9),
        dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8),
        # also include sklearn defaults as a sanity baseline
        dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for params in grid:
        pipe = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=RANDOM_STATE, **params),
                ),
            ]
        )
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        results.append(
            {
                "params": params,
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }
        )
        print(f"  {params}: AUC={scores.mean():.4f} +/- {scores.std():.4f}")

    results.sort(key=lambda r: r["mean"], reverse=True)
    best = results[0]
    audit["hyperparameter_grid_results"] = results
    audit["chosen_config"] = best["params"]
    print(f"BEST: {best['params']}  CV AUC = {best['mean']:.4f}")

    # ---- Refit on full train, predict on test
    final_pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                GradientBoostingClassifier(random_state=RANDOM_STATE, **best["params"]),
            ),
        ]
    )
    final_pipe.fit(X, y)
    test_proba = final_pipe.predict_proba(X_test)[:, 1]

    # ---- Out-of-fold sanity AUC with the chosen config for the headline metric
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        m = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "clf",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE, **best["params"]
                    ),
                ),
            ]
        )
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    oof_auc = float(roc_auc_score(y, oof))
    print(f"OOF AUC (chosen config) = {oof_auc:.4f}")

    # ---- Write submission. Test ids define what the grader needs predictions for;
    # sample_submission.csv has a different id range (1677-2795) that doesn't match
    # the rows in test.csv (15-1665), so we write predictions keyed by test.csv id.
    submission = pd.DataFrame(
        {
            ID_COL: test[ID_COL].values,
            TARGET: test_proba,
        }
    )
    sub_path = HERE / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  ({len(submission)} rows)")

    # ---- Write result.json
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "roc_auc",
        "headline_value": oof_auc,
        "all_metrics": {
            "cv_roc_auc_mean": best["mean"],
            "cv_roc_auc_std": best["std"],
            "oof_roc_auc": oof_auc,
        },
        "skills_applied": [
            "skill_probe_group_structure_in_ids.md",
            "skill_stringify_low_cardinality_int_codes.md",
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Binary classification with mild imbalance (~12% positive). "
            "Sklearn GradientBoostingClassifier with a 4-config grid + 5-fold "
            "StratifiedKFold CV. Low-cardinality int codes (Education, JobLevel, "
            "etc.) were stringified and one-hot encoded so the model treats them "
            "as categories rather than ordinal magnitudes; constant cols "
            "(EmployeeCount, StandardHours, Over18) dropped."
        ),
        "audit": audit,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    res_path = HERE / "result.json"
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {res_path}")


if __name__ == "__main__":
    main()
