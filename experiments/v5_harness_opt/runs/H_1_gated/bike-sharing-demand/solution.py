"""
Bike Sharing Demand — RMSLE regression.

Skills applied:
  - skill_verify_split_claim_before_holdout: prompt claims test = days 20+,
    but train and test both span days 1-19 (random interleaving). Use random KFold.
  - skill_tune_when_n_train_supports_it: n_train=8708 >= 500, no size/latency
    constraint tokens => small explicit grid over HistGradientBoostingRegressor.
  - skill_seed_bag_final_regressor: 3-seed bag (0,1,2) on final fits.

Two-head approach (casual + registered, predict separately, sum) is consistent
with C_warned's winning approach in v1.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/bike-sharing-demand/data")
PROMPT = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/bike-sharing-demand/full_prompt.txt").read_text()

# ----- load
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sub_template = pd.read_csv(DATA / "sampleSubmission.csv")

train["datetime"] = pd.to_datetime(train["datetime"])
test["datetime"] = pd.to_datetime(test["datetime"])

# ----- SKILL 1: verify split claim
audit = {}
tr_dt = train["datetime"]
te_dt = test["datetime"]
audit["train_date_range"] = (str(tr_dt.min()), str(tr_dt.max()))
audit["test_date_range"] = (str(te_dt.min()), str(te_dt.max()))
audit["train_day_max"] = int(tr_dt.dt.day.max())
audit["test_day_max"] = int(te_dt.dt.day.max())
audit["ranges_overlap"] = bool(tr_dt.max() >= te_dt.min() and te_dt.max() >= tr_dt.min())
m = re.search(r"days?\s+(\d+)\s*\+", PROMPT, re.IGNORECASE)
if m:
    claimed_min = int(m.group(1))
    audit["claim_min_day"] = claimed_min
    audit["claim_holds"] = bool(audit["test_day_max"] >= claimed_min and audit["train_day_max"] < claimed_min)
    if not audit["claim_holds"]:
        print(
            f"WARNING: prompt claims test starts at day {claimed_min}, "
            f"actual test_day_max={audit['test_day_max']} train_day_max={audit['train_day_max']}. "
            "Trust data: using random KFold."
        )
audit["cv_choice"] = "KFold(5, shuffle=True, random_state=0)"

# ----- features
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    out = pd.DataFrame({
        "season": df["season"].astype(int),
        "holiday": df["holiday"].astype(int),
        "workingday": df["workingday"].astype(int),
        "weather": df["weather"].astype(int),
        "temp": df["temp"].astype(float),
        "atemp": df["atemp"].astype(float),
        "humidity": df["humidity"].astype(float),
        "windspeed": df["windspeed"].astype(float),
        "year": dt.dt.year.astype(int),
        "month": dt.dt.month.astype(int),
        "day": dt.dt.day.astype(int),
        "hour": dt.dt.hour.astype(int),
        "dayofweek": dt.dt.dayofweek.astype(int),
        "dayofyear": dt.dt.dayofyear.astype(int),
        # cyclic hour
        "hour_sin": np.sin(2 * np.pi * dt.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.dt.hour / 24),
        "month_sin": np.sin(2 * np.pi * dt.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.dt.month / 12),
    })
    return out

X = featurize(train).values
X_test = featurize(test).values
y_count = train["count"].values.astype(float)
y_casual = train["casual"].values.astype(float)
y_registered = train["registered"].values.astype(float)

# RMSLE -> train on log1p, metric on log-space => RMSE on log1p == RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

# ----- SKILL 2: tune when n_train supports it
n_train = len(train)
constraint_tokens = re.findall(r"\b(size|latency|MB|ms|production|mobile|p99)\b", PROMPT, re.IGNORECASE)
should_tune = (n_train >= 500) and (not constraint_tokens)
audit["should_tune"] = should_tune
audit["n_train"] = n_train

# Small grid for HistGradientBoosting (fast on 8k rows, A100 unused — tabular)
grid = [
    dict(max_iter=400, learning_rate=0.05, max_depth=6, min_samples_leaf=20, l2_regularization=0.0),
    dict(max_iter=600, learning_rate=0.03, max_depth=8, min_samples_leaf=20, l2_regularization=0.0),
    dict(max_iter=400, learning_rate=0.05, max_depth=8, min_samples_leaf=10, l2_regularization=1.0),
]

cv = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_two_head(params):
    """OOF RMSLE of predicting count = casual_pred + registered_pred, both on log1p target."""
    oof_total = np.zeros(n_train)
    for tr_idx, va_idx in cv.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        # casual head
        mc = HistGradientBoostingRegressor(**params, random_state=0).fit(Xtr, np.log1p(y_casual[tr_idx]))
        # registered head
        mr = HistGradientBoostingRegressor(**params, random_state=0).fit(Xtr, np.log1p(y_registered[tr_idx]))
        c_pred = np.expm1(mc.predict(Xva))
        r_pred = np.expm1(mr.predict(Xva))
        oof_total[va_idx] = np.clip(c_pred, 0, None) + np.clip(r_pred, 0, None)
    return rmsle(y_count, oof_total)

if should_tune:
    results = []
    for params in grid:
        t0 = time.time()
        score = cv_two_head(params)
        dt_s = time.time() - t0
        print(f"  {params} -> RMSLE={score:.4f}  ({dt_s:.1f}s)")
        results.append((score, params))
    results.sort(key=lambda x: x[0])
    best_score, best_params = results[0]
    audit["hyperparameter_grid"] = [{"params": p, "rmsle_oof": s} for s, p in results]
    audit["chosen_config"] = best_params
    audit["oof_rmsle_chosen"] = best_score
else:
    best_params = dict(max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=20)
    audit["chosen_config"] = best_params
    audit["oof_rmsle_chosen"] = cv_two_head(best_params)
    best_score = audit["oof_rmsle_chosen"]

print(f"chosen config: {best_params}  OOF RMSLE: {best_score:.4f}")

# ----- SKILL 3: seed-bag final fit (3 seeds, two-head)
seeds = [0, 1, 2]
casual_preds = []
registered_preds = []
y_casual_log = np.log1p(y_casual)
y_reg_log = np.log1p(y_registered)
for s in seeds:
    mc = HistGradientBoostingRegressor(**best_params, random_state=s).fit(X, y_casual_log)
    mr = HistGradientBoostingRegressor(**best_params, random_state=s).fit(X, y_reg_log)
    casual_preds.append(np.expm1(mc.predict(X_test)))
    registered_preds.append(np.expm1(mr.predict(X_test)))
casual_mean = np.clip(np.mean(casual_preds, axis=0), 0, None)
reg_mean = np.clip(np.mean(registered_preds, axis=0), 0, None)
test_pred = casual_mean + reg_mean

audit["seed_bag"] = seeds
audit["two_head"] = True

# ----- write submission
sub = pd.DataFrame({
    "datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "count": test_pred,
})
sub.to_csv(HERE / "submission.csv", index=False)
print(f"wrote submission.csv ({len(sub)} rows)")

# ----- result.json
result = {
    "target_column_chosen": "count",
    "headline_metric": "RMSLE",
    "headline_value": float(best_score),
    "all_metrics": {"oof_rmsle": float(best_score)},
    "skills_applied": [
        "skill_verify_split_claim_before_holdout.md",
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "Verified the prompt's 'days 20+' claim against data — both train and test "
        "span days 1-19, so used random KFold(5) instead of a strict temporal holdout. "
        "Two-head approach (casual+registered, log1p targets), HistGradientBoosting tuned "
        "over a 3-config grid, then 3-seed bag on the final fit."
    ),
    "audit": audit,
    "split_claim": "test = days 20+ of each month",
    "claim_verified": bool(audit.get("claim_holds", False)),
    "actual_distribution": {
        "train_day_range": [int(tr_dt.dt.day.min()), int(tr_dt.dt.day.max())],
        "test_day_range": [int(te_dt.dt.day.min()), int(te_dt.dt.day.max())],
        "ranges_overlap": audit["ranges_overlap"],
    },
}
(HERE / "result.json").write_text(json.dumps(result, indent=2, default=str))
print("wrote result.json")
