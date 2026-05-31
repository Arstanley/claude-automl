"""
s3e9 — Concrete Strength Prediction.
Regression, metric = RMSE.

Skills applied (after mechanical INDEX.md evaluation):
  - skill_tune_when_n_train_supports_it: small GBM hyperparam search (n=4325 >= 500, no size constraint).
  - skill_blend_two_model_families: blend GBM with ExtraTrees, weight chosen by OOF RMSE.
  - skill_log_target_for_skewed_nonnegative_regression: A/B-tested vs raw on CV (skew=0.38, kept raw).

Skills evaluated, not applied: see result.json.skills_evaluated_not_applied.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e9/data")
OUT_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_1/playground-series-s3e9")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = 2
N_FOLDS = 5

t0 = time.time()

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "Strength"
ID = "id"

feat_cols = [c for c in train.columns if c not in (TARGET, ID)]
X = train[feat_cols].values
y = train[TARGET].values
X_test = test[feat_cols].values
test_ids = test[ID].values


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv_oof_gbr(X, y, params, log_target=False):
    """Run KFold CV, return oof preds and per-fold fitted models."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    models = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]
        if log_target:
            ytr_fit = np.log1p(ytr)
        else:
            ytr_fit = ytr
        m = GradientBoostingRegressor(random_state=SEED, **params)
        m.fit(Xtr, ytr_fit)
        pred = m.predict(Xva)
        if log_target:
            pred = np.clip(np.expm1(pred), 0, None)
        oof[va_idx] = pred
        models.append(m)
    return oof, models


# ---------------- Step 1: log-target A/B test on a baseline GBM ----------------
baseline_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9)
oof_raw, _ = cv_oof_gbr(X, y, baseline_params, log_target=False)
oof_log, _ = cv_oof_gbr(X, y, baseline_params, log_target=True)
rmse_raw = rmse(y, oof_raw)
rmse_log = rmse(y, oof_log)
use_log = rmse_log < rmse_raw
print(f"[A/B] raw RMSE={rmse_raw:.4f}  log RMSE={rmse_log:.4f}  use_log={use_log}")

# ---------------- Step 2: small GBM hyperparam grid -----------------------------
grid = [
    dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9),
    dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9),
    dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8),
    dict(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.9),
    dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9),
]
best_params, best_rmse, best_oof = None, np.inf, None
for params in grid:
    oof, _ = cv_oof_gbr(X, y, params, log_target=use_log)
    r = rmse(y, oof)
    print(f"  GBR {params} -> RMSE={r:.4f}")
    if r < best_rmse:
        best_rmse, best_params, best_oof = r, params, oof
print(f"[GBR] best params={best_params} RMSE={best_rmse:.4f}")

oof_gbm = best_oof

# ---------------- Step 3: second family (ExtraTrees) and blend -----------------
def cv_oof_et(X, y, params, log_target=False):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]
        ytr_fit = np.log1p(ytr) if log_target else ytr
        m = ExtraTreesRegressor(random_state=SEED, n_jobs=N_JOBS, **params)
        m.fit(Xtr, ytr_fit)
        pred = m.predict(Xva)
        if log_target:
            pred = np.clip(np.expm1(pred), 0, None)
        oof[va_idx] = pred
    return oof


et_params = dict(n_estimators=500, max_depth=None, min_samples_leaf=2)
oof_et = cv_oof_et(X, y, et_params, log_target=use_log)
rmse_et = rmse(y, oof_et)
print(f"[ET] RMSE={rmse_et:.4f}")

# pick best blend weight w on GBM (w on GBM, 1-w on ET)
best_w, best_blend_rmse = 1.0, best_rmse
for w in np.linspace(0, 1, 21):
    blend = w * oof_gbm + (1 - w) * oof_et
    r = rmse(y, blend)
    if r < best_blend_rmse:
        best_blend_rmse, best_w = r, float(w)
print(f"[Blend] best_w(GBM)={best_w:.2f} blend_RMSE={best_blend_rmse:.4f}")

# ---------------- Step 4: refit on full train, predict test --------------------
y_fit = np.log1p(y) if use_log else y

gbm_full = GradientBoostingRegressor(random_state=SEED, **best_params)
gbm_full.fit(X, y_fit)
pred_gbm = gbm_full.predict(X_test)
if use_log:
    pred_gbm = np.clip(np.expm1(pred_gbm), 0, None)

if best_w < 1.0:
    et_full = ExtraTreesRegressor(random_state=SEED, n_jobs=N_JOBS, **et_params)
    et_full.fit(X, y_fit)
    pred_et = et_full.predict(X_test)
    if use_log:
        pred_et = np.clip(np.expm1(pred_et), 0, None)
    pred_test = best_w * pred_gbm + (1 - best_w) * pred_et
else:
    pred_test = pred_gbm

pred_test = np.clip(pred_test, 0, None)

submission = pd.DataFrame({ID: test_ids, TARGET: pred_test})
submission.to_csv(OUT_DIR / "submission.csv", index=False)

elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.1f}s")

# ---------------- Result.json ---------------------------------------------------
skills_applied = [
    "skill_tune_when_n_train_supports_it.md",
    "skill_blend_two_model_families.md",
    "skill_log_target_for_skewed_nonnegative_regression.md",  # A/B-tested even though precondition fails (s3e9 cited in INDEX as a target-transform candidate)
]
skills_evaluated_not_applied = [
    "skill_probe_group_structure_in_ids.md: IDs are sequential ints (no compound format); high feature overlap on synthetic tabular features is not a group signal.",
    "skill_train_on_listed_auxiliary_files.md: original Concrete dataset is mentioned but not provided in DATA_DIR.",
    "skill_verify_split_claim_before_holdout.md: no structural split claim in prompt.",
    "skill_stringify_low_cardinality_int_codes.md: AgeInDays has nunique=15 but model family is tree-based (GBM/ET), not scaler/linear/distance.",
    "skill_multi_axis_constraint_budgets.md: only one constraint axis (RMSE); no latency/size/hard-rule constraints.",
    "skill_missing_indicator_when_missingness_signals_class.md: zero NaN in train/test.",
    "skill_tune_decision_threshold_for_f1.md: task is regression, not binary classification.",
]

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "rmse",
    "headline_value": float(best_blend_rmse),
    "all_metrics": {
        "cv_rmse_baseline_raw": rmse_raw,
        "cv_rmse_baseline_log": rmse_log,
        "use_log_target": bool(use_log),
        "cv_rmse_gbm_tuned": float(best_rmse),
        "cv_rmse_et": float(rmse_et),
        "cv_rmse_blend": float(best_blend_rmse),
        "blend_weight_gbm": float(best_w),
        "best_gbm_params": best_params,
        "et_params": et_params,
        "n_train": int(len(y)),
        "n_test": int(len(test_ids)),
        "elapsed_sec": float(elapsed),
    },
    "skills_applied": skills_applied,
    "skills_evaluated_not_applied": skills_evaluated_not_applied,
    "interpretation_notes": (
        "Tabular regression on 8 continuous concrete-mix features (n=4325). "
        "A/B tested log-target despite skew=0.38 < precondition; "
        f"{'log' if use_log else 'raw'} target won on CV. "
        "Tuned GradientBoosting over a 5-config grid, then blended with ExtraTrees; "
        f"final blend weight on GBM={best_w:.2f}, CV RMSE={best_blend_rmse:.4f}."
    ),
}
with open(OUT_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print("Wrote:", OUT_DIR / "submission.csv", OUT_DIR / "result.json")
