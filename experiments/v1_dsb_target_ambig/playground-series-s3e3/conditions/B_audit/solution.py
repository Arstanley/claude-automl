"""
Playground Series S3E3 — Employee Attrition (synthetic).

Audit-driven design choices (see ambiguity_audit.md):

1. Target column is `Attrition`, chosen because it is the unique column
   present in train.csv but absent from test.csv, AND it is the column
   named in sample_submission.csv's header. Other binary-in-train
   candidates (OverTime, PerformanceRating, Gender) are all present in
   test.csv, so they cannot be the held-out label.
2. ID column is `id` (matches sample_submission.csv header). The prompt's
   `EmployeeNumber` does not exist in either file; the prompt's example
   IDs `1677/1678` are not in test.csv (test ids are 15..1665). We
   ignore the prompt and use the actual test.csv `id` values.
   Note: sample_submission.csv's *id values* (1677..2795) are also
   disjoint from test.csv — the file only matches in header shape, not
   in row identity. We use test.csv ids as the source of truth.
3. Constant columns `EmployeeCount`, `Over18`, `StandardHours` are
   dropped (zero variance, no signal). `id` is dropped from features
   (artifact, potential ordering leakage).
4. Evaluation: stratified 5-fold CV reporting mean+std of ROC-AUC
   (primary) and PR-AUC (color). Submission file contains
   `predict_proba(...)[:, 1]` floats in [0, 1], never hard labels.
5. Sensitive attributes `Gender`, `Age`, `MaritalStatus` are USED as
   features; no fairness mitigation is applied. This is noted in the
   report rather than silently ignored.
6. We do NOT consult the original IBM HR Attrition dataset.

Model: GradientBoostingClassifier with one-hot encoding of nominals.
Ordinal "rating" columns are kept as integers so their ordering is
preserved (no blind one-hot of ordinals).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"

TARGET = "Attrition"
ID = "id"
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]
NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
RANDOM_STATE = 42


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def build_pipeline(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLS,
            ),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    train, test = load()

    # Verify audit-checked facts before proceeding.
    assert TARGET in train.columns and TARGET not in test.columns, (
        f"{TARGET} should be the unique train-only column"
    )
    assert ID in train.columns and ID in test.columns, "id column missing"
    assert set(train[ID]).isdisjoint(set(test[ID])), "train/test id overlap!"
    for c in CONSTANT_COLS:
        assert train[c].nunique() == 1, f"{c} not constant in train"

    drop_cols = CONSTANT_COLS + [ID, TARGET]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    numeric_cols = [c for c in feature_cols if c not in NOMINAL_COLS]

    X = train[feature_cols].copy()
    y = train[TARGET].astype(int).values
    X_test = test[feature_cols].copy()

    pipe = build_pipeline(numeric_cols)

    # 5-fold stratified CV with out-of-fold probabilities for a single
    # robust AUC estimate (small data: 1341 rows).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs: list[float] = []
    fold_aps: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        pipe_f = build_pipeline(numeric_cols)
        pipe_f.fit(X.iloc[tr_idx], y[tr_idx])
        p = pipe_f.predict_proba(X.iloc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[va_idx], p))
        fold_aps.append(average_precision_score(y[va_idx], p))
        print(f"fold {fold}: AUC={fold_aucs[-1]:.4f}  PR-AUC={fold_aps[-1]:.4f}")

    cv_auc_mean = float(np.mean(fold_aucs))
    cv_auc_std = float(np.std(fold_aucs))
    cv_ap_mean = float(np.mean(fold_aps))
    cv_ap_std = float(np.std(fold_aps))

    # Also produce out-of-fold predictions for a single pooled AUC.
    oof = cross_val_predict(
        build_pipeline(numeric_cols), X, y, cv=skf, method="predict_proba", n_jobs=1
    )[:, 1]
    pooled_oof_auc = float(roc_auc_score(y, oof))
    pooled_oof_ap = float(average_precision_score(y, oof))

    print(
        f"CV AUC mean={cv_auc_mean:.4f} std={cv_auc_std:.4f} | "
        f"pooled OOF AUC={pooled_oof_auc:.4f}"
    )
    print(
        f"CV PR-AUC mean={cv_ap_mean:.4f} std={cv_ap_std:.4f} | "
        f"pooled OOF PR-AUC={pooled_oof_ap:.4f}"
    )

    # Refit on all training data and predict on test.
    final = build_pipeline(numeric_cols)
    final.fit(X, y)
    test_proba = final.predict_proba(X_test)[:, 1]
    assert test_proba.shape == (len(test),)
    assert np.all((test_proba >= 0.0) & (test_proba <= 1.0))
    assert np.isfinite(test_proba).all()

    # Submission: header `id,Attrition`, one row per test.csv row,
    # using test.csv id values (not sample_submission's disjoint ids,
    # not the prompt's bogus EmployeeNumber/1677/1678 example).
    submission = pd.DataFrame({ID: test[ID].values, TARGET: test_proba})
    assert len(submission) == len(test)
    sub_path = HERE / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"wrote {sub_path} with {len(submission)} rows")

    result = {
        "target_column_chosen": TARGET,
        "id_column_chosen": ID,
        "target_column_justification": (
            "Attrition is the only column in train.csv but not test.csv, "
            "and is the column named in sample_submission.csv's header. "
            "Other binary-in-train candidates (OverTime, PerformanceRating, "
            "Gender) all appear in test.csv, so they cannot be the held-out "
            "label. Multi-class/ordinal/continuous columns are disqualified "
            "by the AUC-on-binary evaluation rule."
        ),
        "headline_metric": "ROC_AUC_5fold_stratified_cv_mean",
        "headline_value": cv_auc_mean,
        "all_metrics": {
            "cv_roc_auc_mean": cv_auc_mean,
            "cv_roc_auc_std": cv_auc_std,
            "cv_roc_auc_per_fold": fold_aucs,
            "pooled_oof_roc_auc": pooled_oof_auc,
            "cv_pr_auc_mean": cv_ap_mean,
            "cv_pr_auc_std": cv_ap_std,
            "cv_pr_auc_per_fold": fold_aps,
            "pooled_oof_pr_auc": pooled_oof_ap,
            "train_positive_rate": float(np.mean(y)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "audit_items_addressed": [
            "Sec 1: target is Attrition (unique train-only binary col).",
            "Sec 0/3: id column is `id` from test.csv; ignored prompt's "
            "EmployeeNumber/1677/1678 example and sample_submission's "
            "disjoint id range (1677..2795).",
            "Sec 2/4.6: submission is predict_proba()[:,1] floats in [0,1], "
            "not hard 0/1 labels.",
            "Sec 3/4.4: dropped constants EmployeeCount, Over18, "
            "StandardHours and dropped `id` from features.",
            "Sec 3/4.3: stratified 5-fold CV with AUC mean+std reported "
            "instead of single holdout on 1341 rows.",
            "Sec 3: PR-AUC also reported because of class imbalance "
            "(positive rate ~11.6%).",
            "Sec 3/4.7: noted sensitive attributes Gender, Age, "
            "MaritalStatus are used as features with no fairness "
            "mitigation applied.",
            "Sec 3: did NOT consult original IBM HR Attrition dataset; "
            "treated synthetic data as ground truth.",
            "Sec 5: did not submit constant sample_submission; did not "
            "use OverTime/PerformanceRating; did not feed id as a feature.",
        ],
        "interpretation_notes": (
            "The prompt header example (`EmployeeNumber,<target>` with "
            "ids 1677/1678) is wrong: no EmployeeNumber column exists, "
            "and those ids are outside test.csv's id range (15..1665). "
            "sample_submission.csv has the correct *header* (id,Attrition) "
            "but its id values (1677..2795) are disjoint from test.csv. "
            "We follow the audit and use test.csv's actual id column. "
            "Submission has exactly 336 rows. Model: "
            "GradientBoostingClassifier on one-hot-encoded nominals + "
            "passthrough integer ordinals/continuous. Ordinal rating "
            "columns (JobSatisfaction, Education, etc.) are kept as "
            "integers so their ordering is preserved. Sensitive "
            "attributes (Gender, Age, MaritalStatus) are used as features "
            "without fairness mitigation — flagged, not fixed, per the "
            "audit's spirit-checklist item 4.7."
        ),
    }
    res_path = HERE / "result.json"
    with res_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {res_path}")


if __name__ == "__main__":
    main()
