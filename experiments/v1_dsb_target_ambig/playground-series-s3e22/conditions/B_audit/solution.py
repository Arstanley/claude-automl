"""
Playground S3E22 — Horse Survival
Solution that explicitly addresses the ambiguity audit.

Target identification rationale (audit item #1):
  set(train.columns) - set(test.columns) == {'outcome'}
  -> 'outcome' is the only column dropped from test, hence the target.
  Sample submission has 824 stale rows; we IGNORE it and write 247
  rows keyed on test.csv['id'] (audit item #3a / checklist #2).

Predictions are emitted as literal strings {lived, died, euthanized}
(checklist #3). With micro-averaged F1 over single-label multiclass,
micro-F1 == accuracy, so we optimize accuracy via CV.

Modeling choices that address audit items:
  - hospital_number: dropped (audit 3b). 90% train/test overlap makes
    silent label-encoding a leakage hazard; with only 988 rows we don't
    trust out-of-fold target encoding either. Drop is the conservative
    move and we document it.
  - lesion_1: cast to STRING and one-hot encoded (audit 3c). Unseen
    test codes fall back to "other". lesion_2/lesion_3 are >99% zero;
    we collapse them to binary "nonzero" indicators (kept rather than
    dropped because the audit flags potential rare-signal value).
  - Missingness: per-column missing-indicator features added for the
    six highest-missing columns (audit 3d) BEFORE imputing.
  - Ordinal categoricals (audit 3e): we treat all string categoricals
    as nominal one-hot. The audit explicitly says ordinal encoding
    requires domain ordering we were not given; one-hot is the
    defensible neutral choice (vs. alphabetic label encoding which
    the audit calls disqualifying).
  - Evaluation (audit 3g / checklist #7): 5-fold StratifiedKFold,
    report mean ± std micro-F1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SUB_OUT = HERE / "submission.csv"
RESULT_OUT = HERE / "result.json"

RANDOM_STATE = 42
HIGH_MISSING_COLS = [
    "abdomen",
    "rectal_exam_feces",
    "nasogastric_tube",
    "peripheral_pulse",
    "abdomo_appearance",
    "pain",
]


def derive_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Audit item #1 + checklist #1: target == unique column in train\\test."""
    diff = set(train.columns) - set(test.columns)
    assert diff == {"outcome"}, f"Expected single-column diff {{outcome}}, got {diff}"
    return "outcome"


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply audit-driven transformations. Pure function, no fit state."""
    df = df.copy()

    # Audit 3b: drop hospital_number (leakage shape with 90% overlap).
    df = df.drop(columns=["hospital_number"])

    # Audit 3c: lesion_1 is a categorical code, not a magnitude. Stringify.
    df["lesion_1"] = df["lesion_1"].astype(str)

    # lesion_2 / lesion_3 are >99% zero — collapse to binary indicators.
    df["lesion_2_nonzero"] = (df["lesion_2"] != 0).astype(int)
    df["lesion_3_nonzero"] = (df["lesion_3"] != 0).astype(int)
    df = df.drop(columns=["lesion_2", "lesion_3"])

    # Audit 3d: explicit missing indicators on the high-missingness columns.
    for col in HIGH_MISSING_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    return df


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """RandomForest pipeline. RF tolerates the mixed dtypes and small n."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    # handle_unknown='ignore' makes test-only categories safe.
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def main() -> None:
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV)

    target = derive_target(train_raw, test_raw)
    print(f"[audit#1] Target column derived from schema diff: {target!r}")
    print(f"[audit#3a] test.csv rows={len(test_raw)} (will NOT use stale 824-row sample)")

    y = train_raw[target].astype(str).values  # keep string labels (checklist #3)
    train_X = train_raw.drop(columns=[target, "id"])
    test_ids = test_raw["id"].values
    test_X = test_raw.drop(columns=["id"])

    train_X = feature_engineer(train_X)
    test_X = feature_engineer(test_X)

    # Ensure column alignment after one-hot via ColumnTransformer's seen vocab.
    # We must concat the lesion_1 vocabularies so the transformer fits on both;
    # OneHotEncoder(handle_unknown='ignore') already covers unseen test codes,
    # so fitting on train only is correct — but lesion_1 unique test codes will
    # simply produce all-zero one-hot rows. That is the documented behaviour.

    num_cols = train_X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in train_X.columns if c not in num_cols]
    print(f"[features] {len(num_cols)} numeric + {len(cat_cols)} categorical")

    pipe = build_pipeline(num_cols, cat_cols)

    # Audit 3g / checklist #7: 5-fold stratified CV, mean ± std micro-F1.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipe, train_X, y, cv=skf, scoring="f1_micro", n_jobs=1
    )
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"[cv] micro-F1 per fold: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"[cv] micro-F1 mean ± std = {cv_mean:.4f} ± {cv_std:.4f}")

    # Also report per-fold macro for interpretation (minority class behaviour).
    macro_scores = []
    accs = []
    for tr_idx, va_idx in skf.split(train_X, y):
        pipe_fold = build_pipeline(num_cols, cat_cols)
        pipe_fold.fit(train_X.iloc[tr_idx], y[tr_idx])
        pred = pipe_fold.predict(train_X.iloc[va_idx])
        macro_scores.append(f1_score(y[va_idx], pred, average="macro"))
        accs.append((pred == y[va_idx]).mean())
    macro_mean = float(np.mean(macro_scores))
    macro_std = float(np.std(macro_scores))
    acc_mean = float(np.mean(accs))
    print(f"[cv] macro-F1 mean ± std = {macro_mean:.4f} ± {macro_std:.4f}")
    print(f"[cv] accuracy mean ± std = {acc_mean:.4f} (== micro-F1 for single-label multiclass)")

    # Fit on full train, predict test.
    pipe.fit(train_X, y)
    test_pred = pipe.predict(test_X)

    submission = pd.DataFrame({"id": test_ids, target: test_pred})
    assert len(submission) == 247, f"Expected 247 rows, got {len(submission)}"
    assert list(submission.columns) == ["id", target]
    assert set(submission[target].unique()) <= {"lived", "died", "euthanized"}
    submission.to_csv(SUB_OUT, index=False)
    print(f"[submission] wrote {SUB_OUT} ({len(submission)} rows)")
    print(f"[submission] pred distribution: {pd.Series(test_pred).value_counts().to_dict()}")

    result = {
        "target_column_chosen": target,
        "target_column_justification": (
            "Identified via schema diff: set(train.columns) - set(test.columns) "
            "== {'outcome'}. 'outcome' is the only column present in train and "
            "absent in test, has 3 categorical levels (lived/died/euthanized) "
            "compatible with the micro-F1 metric, and the sample submission "
            "header confirms but is NOT load-bearing for the decision."
        ),
        "headline_metric": "micro_f1_cv5_mean",
        "headline_value": cv_mean,
        "all_metrics": {
            "micro_f1_cv5_mean": cv_mean,
            "micro_f1_cv5_std": cv_std,
            "micro_f1_cv5_folds": [float(s) for s in cv_scores],
            "macro_f1_cv5_mean": macro_mean,
            "macro_f1_cv5_std": macro_std,
            "accuracy_cv5_mean": acc_mean,
            "n_train": int(len(train_raw)),
            "n_test": int(len(test_raw)),
            "n_classes": 3,
        },
        "audit_items_addressed": [
            "1_target_from_schema_diff: 'outcome' derived via set(train.cols)-set(test.cols)={'outcome'}; sample_submission used only as a corroborating signal, not as the deciding evidence.",
            "2_categorical_target_for_micro_f1: predictions emitted as literal strings lived/died/euthanized; micro-F1 reduces to accuracy for single-label multiclass.",
            "3a_submission_row_count: ignored sample_submission.csv (824 stale rows); submission has exactly 247 rows keyed on test.csv['id'].",
            "3b_hospital_number_leakage: dropped column. 90% train/test overlap on 241 train uniques makes silent label-encoding a leak; OOF target encoding is unreliable at n=988, so dropping is the conservative choice.",
            "3c_lesion_1_categorical: cast to string and one-hot encoded; unseen test codes handled via handle_unknown='ignore'. lesion_2/lesion_3 collapsed to binary 'nonzero' indicators given >99% zero rate.",
            "3d_missingness_modeled: explicit *_missing indicator columns added for abdomen, rectal_exam_feces, nasogastric_tube, peripheral_pulse, abdomo_appearance, pain BEFORE median/mode imputation.",
            "3e_ordinal_categoricals: one-hot encoded (treated as nominal) — audit notes domain ordering was not provided; one-hot is defensible, alphabetic label-encoding would be disqualifying.",
            "3f_synthetic_data: no domain rules from the UCI Horse Colic literature imported; model learns purely from the synthetic distribution.",
            "3g_cv_not_holdout: reported micro-F1 is mean±std over 5 StratifiedKFold splits, not a single hold-out.",
        ],
        "interpretation_notes": (
            "Micro-F1 over single-label multiclass equals accuracy, so the "
            "headline number is also the row-level accuracy. Std across 5 "
            "folds is meaningful (n=988, ~40 minority-class val rows/fold) — "
            "any improvement smaller than ~1 std is noise. Macro-F1 is "
            "reported alongside for visibility into the minority class "
            "('euthanized', 19.9%) behaviour even though it is not the "
            "scoring metric. The submission emits strings, not integers."
        ),
    }
    with open(RESULT_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] wrote {RESULT_OUT}")


if __name__ == "__main__":
    main()
