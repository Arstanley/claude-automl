"""Playground Series S3E2 — Stroke probability (binary classification, ROC-AUC).

Skills applied:
  * skill_tune_when_n_train_supports_it — small grid on HistGradientBoosting (n_train=12243, no constraints).
  * skill_blend_two_model_families — blend HGB with RandomForest; weight chosen by OOF AUC.

Skills evaluated and not applied: see result.json.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e2/data")
WORK_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1/playground-series-s3e2")
WORK_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
SEED = 42
N_SPLITS = 5

CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
NUM_COLS = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "stroke"
ID_COL = "id"


def build_xy(train: pd.DataFrame, test: pd.DataFrame):
    # Ordinal-encode categoricals so they are tree-friendly and HGB can use `categorical_features`.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    Xtr = train[FEATURES].copy()
    Xte = test[FEATURES].copy()
    Xtr[CAT_COLS] = enc.transform(Xtr[CAT_COLS])
    Xte[CAT_COLS] = enc.transform(Xte[CAT_COLS])
    cat_mask = [c in CAT_COLS for c in FEATURES]
    return Xtr.values, Xte.values, cat_mask


def cv_auc_hgb(X, y, cat_mask, params, n_splits=N_SPLITS) -> tuple[float, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        clf = HistGradientBoostingClassifier(
            categorical_features=cat_mask,
            random_state=SEED,
            **params,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return roc_auc_score(y, oof), oof


def cv_auc_rf(X, y, params, n_splits=N_SPLITS) -> tuple[float, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            random_state=SEED,
            n_jobs=N_JOBS,
            **params,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return roc_auc_score(y, oof), oof


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train[TARGET].values.astype(int)
    X, Xte, cat_mask = build_xy(train, test)

    # -------- HGB: small 4-config grid (skill_tune_when_n_train_supports_it) --------
    hgb_grid = [
        {"max_iter": 400, "learning_rate": 0.05, "max_depth": 4, "l2_regularization": 0.0, "early_stopping": False},
        {"max_iter": 600, "learning_rate": 0.03, "max_depth": 5, "l2_regularization": 0.0, "early_stopping": False},
        {"max_iter": 300, "learning_rate": 0.08, "max_depth": 3, "l2_regularization": 1.0, "early_stopping": False},
        {"max_iter": 500, "learning_rate": 0.05, "max_depth": None, "max_leaf_nodes": 31, "l2_regularization": 0.5, "early_stopping": False},
    ]
    hgb_results = []
    for p in hgb_grid:
        auc, oof = cv_auc_hgb(X, y, cat_mask, p)
        hgb_results.append((auc, p, oof))
        print(f"[hgb] {p} -> AUC {auc:.5f}  (elapsed {time.time()-t0:.1f}s)")
    hgb_results.sort(key=lambda r: r[0], reverse=True)
    best_hgb_auc, best_hgb_params, oof_hgb = hgb_results[0]
    print(f"[hgb] best: {best_hgb_params} AUC={best_hgb_auc:.5f}")

    # -------- RF blend partner (skill_blend_two_model_families) --------
    rf_params = {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5}
    rf_auc, oof_rf = cv_auc_rf(X, y, rf_params)
    print(f"[rf]  AUC {rf_auc:.5f}  (elapsed {time.time()-t0:.1f}s)")

    # Weight search on OOF
    best_w, best_blend = 1.0, best_hgb_auc
    for w in np.linspace(0.0, 1.0, 21):
        b = roc_auc_score(y, w * oof_hgb + (1 - w) * oof_rf)
        if b > best_blend:
            best_blend, best_w = b, float(w)
    print(f"[blend] w_hgb={best_w:.2f}  AUC={best_blend:.5f}")

    # -------- Final fits on full data --------
    hgb = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=SEED,
        **best_hgb_params,
    )
    hgb.fit(X, y)
    p_hgb = hgb.predict_proba(Xte)[:, 1]

    if best_w in (0.0, 1.0):
        # Ship single model — blend partner did not add value.
        if best_w == 1.0:
            p_te = p_hgb
            chosen = "hgb_only"
            headline = best_hgb_auc
        else:
            rf = RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **rf_params)
            rf.fit(X, y)
            p_te = rf.predict_proba(Xte)[:, 1]
            chosen = "rf_only"
            headline = rf_auc
    else:
        rf = RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **rf_params)
        rf.fit(X, y)
        p_rf = rf.predict_proba(Xte)[:, 1]
        p_te = best_w * p_hgb + (1 - best_w) * p_rf
        chosen = "blend"
        headline = best_blend

    # -------- Submission --------
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: p_te})
    sub_path = WORK_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "roc_auc",
        "headline_value": float(headline),
        "all_metrics": {
            "cv_auc_hgb_best": float(best_hgb_auc),
            "cv_auc_rf": float(rf_auc),
            "cv_auc_blend": float(best_blend),
            "blend_weight_hgb": float(best_w),
            "final_choice": chosen,
            "best_hgb_params": {k: (None if v is None else v) for k, v in best_hgb_params.items()},
            "rf_params": rf_params,
            "n_splits": N_SPLITS,
            "n_train": int(len(y)),
            "n_test": int(len(test)),
            "wall_seconds": round(time.time() - t0, 1),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_blend_two_model_families.md",
        ],
        "skills_evaluated_not_applied": [
            "skill_probe_group_structure_in_ids.md: no id overlap train/test, ids are plain ints with no compound structure.",
            "skill_train_on_listed_auxiliary_files.md: prompt mentions original Stroke Prediction Dataset is 'feel free to use', but no such file in DATA_DIR.",
            "skill_verify_split_claim_before_holdout.md: prompt makes no structural (temporal/group) split claim.",
            "skill_stringify_low_cardinality_int_codes.md: only low-card ints are hypertension/heart_disease (already 0/1) and primary model is tree-based, not scaler/linear/distance.",
            "skill_multi_axis_constraint_budgets.md: prompt names no latency/size/accuracy constraints.",
            "skill_missing_indicator_when_missingness_signals_class.md: zero NaNs in train and test.",
            "skill_tune_decision_threshold_for_f1.md: metric is ROC-AUC (rank-based), not F1/F-beta/MCC.",
            "skill_log_target_for_skewed_nonnegative_regression.md: task is classification, not regression.",
        ],
        "interpretation_notes": (
            "Binary stroke prediction with ~4% positive rate; metric is ROC-AUC so probabilities are kept raw (no threshold tuning). "
            "HistGradientBoosting with native categorical handling was tuned over a 4-config grid (5-fold StratifiedKFold), "
            f"then OOF-blended with a RandomForest partner; final choice: {chosen}."
        ),
    }
    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {sub_path} (rows={len(sub)})")
    print(f"wrote {WORK_DIR / 'result.json'}")
    print(f"headline AUC = {headline:.5f}  chosen={chosen}  wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
