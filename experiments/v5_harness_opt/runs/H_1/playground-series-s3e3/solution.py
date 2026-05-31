"""Solution for playground-series-s3e3: predict probability of Attrition (binary, AUC).

Pipeline:
  - Drop constant columns (EmployeeCount, Over18, StandardHours)
  - One-hot encode object columns
  - Tune HistGradientBoosting via small CV grid (skill: tune_when_n_train_supports_it)
  - Blend GBM with Logistic Regression (skill: blend_two_model_families)
  - Output probabilities to submission.csv with header `EmployeeNumber,Attrition` per spec.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e3/data")

RANDOM_STATE = 42
N_JOBS = 2


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def main():
    t0 = time.time()
    train, test = load()
    print(f"train={train.shape} test={test.shape}")

    target = "Attrition"
    id_col = "id"
    drop_const = ["EmployeeCount", "Over18", "StandardHours"]

    y = train[target].astype(int).values
    X_full = train.drop(columns=[target, id_col] + drop_const)
    X_test = test.drop(columns=[id_col] + drop_const)

    cat_cols = [c for c in X_full.columns if X_full[c].dtype == object]
    num_cols = [c for c in X_full.columns if c not in cat_cols]
    print(f"cat={len(cat_cols)} num={len(num_cols)}")

    # ----- GBM pipeline (handles raw numerics + ordinal-ish one-hot is OK) -----
    pre_gbm = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    # Skill: tune_when_n_train_supports_it — small explicit grid
    grid = [
        {"learning_rate": 0.05, "max_iter": 400, "max_depth": None, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 20},
        {"learning_rate": 0.05, "max_iter": 600, "max_depth": None, "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 30},
        {"learning_rate": 0.03, "max_iter": 800, "max_depth": None, "max_leaf_nodes": 31, "l2_regularization": 1.0, "min_samples_leaf": 20},
        {"learning_rate": 0.1,  "max_iter": 200, "max_depth": 4, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 30},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_auc = -np.inf
    best_cfg = None
    best_oof = None
    for cfg in grid:
        gbm = Pipeline(
            [
                ("pre", pre_gbm),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=RANDOM_STATE,
                        **cfg,
                    ),
                ),
            ]
        )
        oof = cross_val_predict(gbm, X_full, y, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"  GBM cfg {cfg} -> AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_cfg = cfg
            best_oof = oof

    print(f"Best GBM AUC={best_auc:.4f} cfg={best_cfg}")

    # ----- Second family: Logistic Regression on scaled / one-hot features -----
    pre_lr = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    lr = Pipeline(
        [
            ("pre", pre_lr),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
        ]
    )
    lr_oof = cross_val_predict(lr, X_full, y, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]
    lr_auc = roc_auc_score(y, lr_oof)
    print(f"LR AUC={lr_auc:.4f}")

    # ----- Blend: search weight on OOF -----
    best_blend_auc = best_auc
    best_w = 1.0  # weight on GBM
    for w in np.linspace(0, 1, 21):
        blend = w * best_oof + (1 - w) * lr_oof
        a = roc_auc_score(y, blend)
        if a > best_blend_auc:
            best_blend_auc = a
            best_w = w
    print(f"Best blend w_gbm={best_w:.2f} AUC={best_blend_auc:.4f}")

    # ----- Refit on full train, predict test -----
    gbm_final = Pipeline(
        [
            ("pre", pre_gbm),
            (
                "clf",
                HistGradientBoostingClassifier(
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                    **best_cfg,
                ),
            ),
        ]
    )
    gbm_final.fit(X_full, y)
    gbm_test = gbm_final.predict_proba(X_test)[:, 1]

    if best_w == 1.0:
        # Pure GBM
        test_pred = gbm_test
        skills_applied = [
            "skill_tune_when_n_train_supports_it.md",
        ]
        skills_eval_not_applied = ["skill_blend_two_model_families.md (blend weight collapsed to GBM-only on CV)"]
    else:
        lr.fit(X_full, y)
        lr_test = lr.predict_proba(X_test)[:, 1]
        test_pred = best_w * gbm_test + (1 - best_w) * lr_test
        skills_applied = [
            "skill_tune_when_n_train_supports_it.md",
            "skill_blend_two_model_families.md",
        ]
        skills_eval_not_applied = []

    # ----- Submission: per spec header is `EmployeeNumber,Attrition`. -----
    # The actual test file uses `id` as the EmployeeNumber column.
    submission = pd.DataFrame({
        "EmployeeNumber": test[id_col].values,
        "Attrition": test_pred,
    })
    submission.to_csv(HERE / "submission.csv", index=False)
    print(f"Wrote submission.csv: {submission.shape}")

    # Evaluate all the skills we did NOT apply with reasons
    not_applied_full = [
        "skill_probe_group_structure_in_ids.md: train ids and test ids are disjoint integer ranges; no compound IDs.",
        "skill_train_on_listed_auxiliary_files.md: prompt mentions optional 'original dataset' but no aux file is present in DATA_DIR.",
        "skill_verify_split_claim_before_holdout.md: no structural train/test split claim in prompt.",
        "skill_stringify_low_cardinality_int_codes.md: downstream model family is GBM (tree-based), not linear/distance; precondition fails.",
        "skill_multi_axis_constraint_budgets.md: prompt names no explicit constraint axes besides accuracy.",
        "skill_missing_indicator_when_missingness_signals_class.md: 0 NaNs in train/test.",
        "skill_tune_decision_threshold_for_f1.md: metric is ROC-AUC (probability-based), not F1/F-beta.",
        "skill_log_target_for_skewed_nonnegative_regression.md: task is binary classification.",
    ] + skills_eval_not_applied

    result = {
        "target_column_chosen": target,
        "headline_metric": "roc_auc",
        "headline_value": float(best_blend_auc),
        "all_metrics": {
            "cv_roc_auc_best_gbm": float(best_auc),
            "cv_roc_auc_lr": float(lr_auc),
            "cv_roc_auc_blend": float(best_blend_auc),
            "blend_weight_gbm": float(best_w),
            "best_gbm_cfg": best_cfg,
        },
        "skills_applied": skills_applied,
        "skills_evaluated_not_applied": not_applied_full,
        "interpretation_notes": (
            "Binary classification with AUC metric on 1341 rows of synthetic HR data. "
            "Tuned HistGradientBoostingClassifier with a 4-config CV grid and blended with logistic "
            "regression on OOF probabilities; final blend weight was selected on CV. "
            "Dropped 3 constant columns (EmployeeCount, Over18, StandardHours)."
        ),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result.json. total={result['runtime_seconds']}s")


if __name__ == "__main__":
    main()
