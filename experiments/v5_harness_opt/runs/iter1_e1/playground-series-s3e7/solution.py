"""Playground Series S3E7 - Reservation Cancellation prediction.

Binary classification, AUC metric.
Skills applied:
- skill_tune_when_n_train_supports_it: n_train=33680 >> 500, no size/latency tokens
  in prompt -> small 3-config GBM grid with 5-fold CV.
- skill_stringify_low_cardinality_int_codes: int-coded categoricals
  (type_of_meal_plan, room_type_reserved, market_segment_type, etc.) are kept
  as-is for tree models (which handle integer categoricals natively) but the
  skill's spirit -> we don't standardize/scale them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e7/data")
N_JOBS = 2
SEED = 42

t0 = time.time()
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

TARGET = "booking_status"
ID = "id"

# Light, model-agnostic feature engineering
def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["price_per_guest"] = df["avg_price_per_room"] / df["total_guests"].clip(lower=1)
    df["price_per_night"] = df["avg_price_per_room"]  # already per-night
    # date-as-ordinal (clip out-of-range like Feb 30 -> NaN, then fillna)
    dt = pd.to_datetime(
        dict(year=df["arrival_year"], month=df["arrival_month"], day=df["arrival_date"]),
        errors="coerce",
    )
    df["arrival_dow"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["arrival_doy"] = dt.dt.dayofyear.fillna(-1).astype(int)
    df["bad_date"] = dt.isna().astype(int)
    df["had_prior_cancel"] = (df["no_of_previous_cancellations"] > 0).astype(int)
    return df


train_fe = add_feats(train)
test_fe = add_feats(test)

feature_cols = [c for c in train_fe.columns if c not in (TARGET, ID)]
X = train_fe[feature_cols].values
y = train_fe[TARGET].values
X_test = test_fe[feature_cols].values

# Small grid: 3 configs covering shallower/deeper/wider trees, slow lr.
configs = [
    dict(max_iter=400, learning_rate=0.05, max_depth=6,  max_leaf_nodes=31,  l2_regularization=0.0),
    dict(max_iter=600, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=1.0),
    dict(max_iter=500, learning_rate=0.07, max_depth=8,  max_leaf_nodes=31,  l2_regularization=0.5),
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = []
best_cfg = None
best_auc = -np.inf
oof_per_cfg = []
for ci, cfg in enumerate(configs):
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = HistGradientBoostingClassifier(
            **cfg,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=SEED + fold,
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    auc = roc_auc_score(y, oof)
    cv_results.append({"cfg": cfg, "auc": float(auc)})
    oof_per_cfg.append(oof)
    print(f"cfg {ci}: AUC={auc:.5f}  ({cfg})  elapsed={time.time()-t0:.1f}s", flush=True)
    if auc > best_auc:
        best_auc = auc
        best_cfg = cfg
        best_ci = ci

print(f"Best cfg #{best_ci}: AUC={best_auc:.5f}", flush=True)

# Refit best config on full train; ensemble 3 seeds for stability.
test_preds = []
for seed in [SEED, SEED + 1, SEED + 2]:
    model = HistGradientBoostingClassifier(
        **best_cfg,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(X, y)
    test_preds.append(model.predict_proba(X_test)[:, 1])
test_pred = np.mean(test_preds, axis=0)

sub = pd.DataFrame({ID: test_fe[ID].values, TARGET: test_pred})
sub.to_csv(HERE / "submission.csv", index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "roc_auc",
    "headline_value": best_auc,
    "all_metrics": {
        "cv_auc_per_cfg": [r["auc"] for r in cv_results],
        "best_cfg_index": best_ci,
        "best_cfg": best_cfg,
        "n_train": int(len(y)),
        "n_test": int(len(test_fe)),
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_seed_bag_final_regressor.md",
    ],
    "interpretation_notes": (
        "HistGradientBoostingClassifier on 33.7k rows of synthetic reservation "
        "data. Tuned via a 3-config grid with 5-fold StratifiedKFold; best "
        f"OOF AUC={best_auc:.4f}. Final test predictions are a 3-seed bag of "
        "the best config refit on full train. sample_submission.csv ID range "
        "doesn't match test.csv -> submitted using test.csv IDs as the spec "
        "instructs."
    ),
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Wrote submission.csv ({len(sub)} rows) and result.json. Total time={time.time()-t0:.1f}s")
