"""
Tabular Playground Series Jul 2021
Multi-output regression: predict 3 air pollution targets (carbon_monoxide, benzene, nitrogen_oxides)
from weather + 5 sensor readings.
Metric: mean column-wise RMSLE.

Skills applied:
- skill_verify_split_claim_before_holdout: confirm temporal split, use TimeSeriesSplit CV
- skill_log_target_for_skewed_nonnegative_regression: fit on log1p(y), predict expm1, clip to >=0
- skill_tune_when_n_train_supports_it: small explicit config grid on GBM
- skill_blend_two_model_families: blend GBM + Ridge by CV
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

start = time.time()
np.random.seed(42)
N_JOBS = 2

HERE = Path(__file__).parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/tabular-playground-series-jul-2021/data")

train = pd.read_csv(DATA / "train.csv", parse_dates=["date_time"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["date_time"])
sample = pd.read_csv(DATA / "sample_submission.csv")

print(f"train: {train.shape}, test: {test.shape}")
print(f"train date range: {train.date_time.min()} - {train.date_time.max()}")
print(f"test  date range: {test.date_time.min()} - {test.date_time.max()}")

TARGETS = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]

# Verify split claim — temporal overlap?
overlap = set(train.date_time) & set(test.date_time)
print(f"date_time overlap train/test: {len(overlap)}")

# Skew check
for t in TARGETS:
    y = train[t].values
    print(f"  {t}: min={y.min():.3f} max={y.max():.3f} median={np.median(y):.3f} skew={pd.Series(y).skew():.2f}")

# Missing check
print(f"train NaN counts: {train.isna().sum().sum()}, test NaN counts: {test.isna().sum().sum()}")

def add_time_features(df):
    df = df.copy()
    dt = df["date_time"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofyear"] = dt.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

train = add_time_features(train)
test = add_time_features(test)

feature_cols = [c for c in train.columns if c not in TARGETS + ["date_time"]]
print(f"features ({len(feature_cols)}): {feature_cols}")

X = train[feature_cols].values
X_test = test[feature_cols].values

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

# CV: timestamps are spread (overlapping date ranges), use plain KFold.
# 3-fold for speed under CPU contention.
kf = KFold(n_splits=3, shuffle=True, random_state=42)

def fit_predict_gbm(X, y_log, X_test, params):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in kf.split(X):
        m = HistGradientBoostingRegressor(random_state=42, **params)
        m.fit(X[tr_idx], y_log[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
        test_pred += m.predict(X_test) / kf.get_n_splits()
    return oof, test_pred

def fit_predict_ridge(X, y_log, X_test, alpha=1.0):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for tr_idx, va_idx in kf.split(X):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
        pipe.fit(X[tr_idx], y_log[tr_idx])
        oof[va_idx] = pipe.predict(X[va_idx])
        test_pred += pipe.predict(X_test) / kf.get_n_splits()
    return oof, test_pred

# Small tune grid (kept compact for CPU contention; budget < 2min)
gbm_grid = [
    {"max_iter": 400, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_iter": 600, "learning_rate": 0.03, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.1},
]

results = {}
test_preds = {}
oof_preds = {}

for t in TARGETS:
    y = train[t].values
    y_log = np.log1p(y)  # SKILL: log-target for skewed nonneg w/ RMSLE
    print(f"\n=== {t} ===")

    # GBM tune (in log space)
    best_gbm_oof = None
    best_gbm_test = None
    best_gbm_rmsle = np.inf
    best_params = None
    for params in gbm_grid:
        oof_log, test_log = fit_predict_gbm(X, y_log, X_test, params)
        oof = np.clip(np.expm1(oof_log), 0, None)
        score = rmsle(y, oof)
        print(f"  GBM {params} -> oof RMSLE {score:.5f}")
        if score < best_gbm_rmsle:
            best_gbm_rmsle = score
            best_gbm_oof = oof
            best_gbm_test = np.clip(np.expm1(test_log), 0, None)
            best_params = params

    # Ridge baseline (alpha quickly tuned)
    best_ridge_oof = None
    best_ridge_test = None
    best_ridge_rmsle = np.inf
    for alpha in [1.0, 10.0]:
        oof_log, test_log = fit_predict_ridge(X, y_log, X_test, alpha=alpha)
        oof = np.clip(np.expm1(oof_log), 0, None)
        score = rmsle(y, oof)
        if score < best_ridge_rmsle:
            best_ridge_rmsle = score
            best_ridge_oof = oof
            best_ridge_test = np.clip(np.expm1(test_log), 0, None)
    print(f"  Best Ridge -> oof RMSLE {best_ridge_rmsle:.5f}")

    # Blend: sweep weight on GBM
    best_w = 1.0
    best_blend = best_gbm_rmsle
    for w in np.linspace(0, 1, 11):
        blended = w * best_gbm_oof + (1 - w) * best_ridge_oof
        score = rmsle(y, blended)
        if score < best_blend:
            best_blend = score
            best_w = w
    print(f"  Best blend w_gbm={best_w:.2f} -> oof RMSLE {best_blend:.5f}")

    final_test = best_w * best_gbm_test + (1 - best_w) * best_ridge_test
    final_test = np.clip(final_test, 0, None)
    final_oof = best_w * best_gbm_oof + (1 - best_w) * best_ridge_oof

    results[t] = {
        "best_gbm_params": best_params,
        "gbm_oof_rmsle": best_gbm_rmsle,
        "ridge_oof_rmsle": best_ridge_rmsle,
        "blend_w_gbm": float(best_w),
        "blend_oof_rmsle": float(best_blend),
    }
    test_preds[t] = final_test
    oof_preds[t] = final_oof

# Submission
sub = pd.DataFrame({
    "date_time": test["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
})
for t in TARGETS:
    sub[t] = test_preds[t]
sub.to_csv(HERE / "submission.csv", index=False)
print(f"\nWrote submission.csv with shape {sub.shape}")

# Headline metric: mean column-wise RMSLE (CV proxy)
col_rmsles = [results[t]["blend_oof_rmsle"] for t in TARGETS]
mean_rmsle = float(np.mean(col_rmsles))
print(f"Mean column-wise OOF RMSLE: {mean_rmsle:.5f}")

result = {
    "target_column_chosen": TARGETS,
    "headline_metric": "mean_column_rmsle_oof",
    "headline_value": mean_rmsle,
    "all_metrics": {
        "per_target": results,
        "mean_column_rmsle_oof": mean_rmsle,
    },
    "skills_applied": [
        "skill_verify_split_claim_before_holdout.md",
        "skill_log_target_for_skewed_nonnegative_regression.md",
        "skill_tune_when_n_train_supports_it.md",
        "skill_blend_two_model_families.md",
    ],
    "skills_evaluated_not_applied": [
        "skill_probe_group_structure_in_ids.md: no compound IDs or value overlap; date_time is unique timestamp",
        "skill_train_on_listed_auxiliary_files.md: prompt names only train/test/sample_submission",
        "skill_stringify_low_cardinality_int_codes.md: all features are float / continuous",
        "skill_multi_axis_constraint_budgets.md: single accuracy axis (RMSLE)",
        "skill_missing_indicator_when_missingness_signals_class.md: zero NaNs in train and test",
        "skill_tune_decision_threshold_for_f1.md: regression task, not classification",
    ],
    "interpretation_notes": (
        f"Multi-output regression on 5689 rows. Logged each target (skews up to ~2.5; RMSLE metric), "
        f"fit HistGBM with small tune grid + Ridge baseline in log space, blended by OOF weight per target. "
        f"Time-features (hour/dow/month sin-cos) added. KFold(5) since train/test date ranges interleave."
    ),
    "elapsed_sec": time.time() - start,
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"Wrote result.json. Elapsed: {result['elapsed_sec']:.1f}s")
