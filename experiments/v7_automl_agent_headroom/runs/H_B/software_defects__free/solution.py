"""
Software Defects binary classification — F1 metric.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=81410 >> 500, GBM family, no
  size/latency/production tokens in prompt → small explicit 3-config grid.

Additional standard moves (not in skill library):
- F1 on imbalanced binary (~22.8% positives): tune decision threshold on
  out-of-fold predictions rather than using 0.5.
- Try lightgbm; fall back to HistGradientBoostingClassifier.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/software_defects/data")
N_JOBS = 2
RANDOM_STATE = 42
N_SPLITS = 5

t0 = time.time()

# --- Load data ---
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "defects"
ID = "id"
features = [c for c in train.columns if c not in (ID, TARGET)]

X = train[features].values
y = train[TARGET].astype(int).values
X_test = test[features].values
test_ids = test[ID].values

print(f"[load] train={X.shape} test={X_test.shape} pos_rate={y.mean():.4f}")

# --- Try LightGBM; fall back to HistGradientBoosting ---
USE_LGB = False
try:
    import lightgbm as lgb  # type: ignore

    USE_LGB = True
    print("[model] using LightGBM")
except Exception as e:
    print(f"[model] LightGBM unavailable ({e}); using HistGradientBoostingClassifier")
    from sklearn.ensemble import HistGradientBoostingClassifier


# --- Configs (small grid per skill_tune_when_n_train_supports_it) ---
if USE_LGB:
    CONFIGS = [
        dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20,
             feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5, reg_lambda=1.0),
        dict(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=40,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_lambda=2.0),
        dict(n_estimators=400, learning_rate=0.08, num_leaves=15, min_child_samples=20,
             feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, reg_lambda=0.5),
    ]
else:
    CONFIGS = [
        dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
        dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=2.0),
        dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.5),
    ]


def fit_predict_oof(cfg):
    """Stratified CV: returns (oof_proba, test_proba_mean)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    test_acc = np.zeros(len(X_test), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if USE_LGB:
            model = lgb.LGBMClassifier(
                **cfg,
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbose=-1,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_acc += model.predict_proba(X_test)[:, 1] / N_SPLITS
        else:
            model = HistGradientBoostingClassifier(
                **cfg,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
            )
            model.fit(X_tr, y_tr)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_acc += model.predict_proba(X_test)[:, 1] / N_SPLITS
    return oof, test_acc


def best_threshold(y_true, proba):
    """Sweep thresholds on a coarse-then-fine grid; return (thr, f1)."""
    thrs = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.quantile(proba, np.linspace(0.05, 0.95, 91))]))
    best_t, best_f1 = 0.5, -1.0
    for t in thrs:
        f1 = f1_score(y_true, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# --- Train each config, track best by OOF F1 (with tuned threshold) ---
results = []
oof_store = {}
test_store = {}
for i, cfg in enumerate(CONFIGS):
    if time.time() - t0 > 150:  # keep budget under ~2.5 min before threshold/blend
        print(f"[budget] skipping config {i}, elapsed={time.time()-t0:.1f}s")
        break
    t1 = time.time()
    oof, tp = fit_predict_oof(cfg)
    thr, f1 = best_threshold(y, oof)
    elapsed = time.time() - t1
    print(f"[cfg {i}] oof_f1@best_thr={f1:.5f} thr={thr:.3f} elapsed={elapsed:.1f}s cfg={cfg}")
    results.append((i, thr, f1, cfg))
    oof_store[i] = oof
    test_store[i] = tp

# --- Blend (simple average) of top configs ---
results_sorted = sorted(results, key=lambda r: r[2], reverse=True)
top_k = min(len(results_sorted), 3)
blend_oof = np.mean([oof_store[r[0]] for r in results_sorted[:top_k]], axis=0)
blend_test = np.mean([test_store[r[0]] for r in results_sorted[:top_k]], axis=0)
blend_thr, blend_f1 = best_threshold(y, blend_oof)
print(f"[blend top-{top_k}] oof_f1={blend_f1:.5f} thr={blend_thr:.3f}")

# --- Choose final: blend vs best single ---
best_single = results_sorted[0]
if blend_f1 > best_single[2]:
    final_proba = blend_test
    final_thr = blend_thr
    final_f1 = blend_f1
    final_label = f"blend_top{top_k}"
else:
    final_proba = test_store[best_single[0]]
    final_thr = best_single[1]
    final_f1 = best_single[2]
    final_label = f"cfg_{best_single[0]}"

print(f"[final] selection={final_label} oof_f1={final_f1:.5f} thr={final_thr:.3f}")

# --- Submission ---
preds = (final_proba >= final_thr).astype(int)
sub = pd.DataFrame({ID: test_ids, "defects": preds})
sub.to_csv(HERE / "submission.csv", index=False)
print(f"[submit] wrote submission.csv pos_rate={preds.mean():.4f}")

# --- result.json ---
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "f1",
    "headline_value": float(final_f1),
    "all_metrics": {
        "oof_f1_per_config": [{"cfg_idx": r[0], "thr": r[1], "oof_f1": r[2]} for r in results],
        "blend_oof_f1": float(blend_f1),
        "blend_thr": float(blend_thr),
        "final_selection": final_label,
        "final_threshold": float(final_thr),
        "test_positive_rate": float(preds.mean()),
        "train_positive_rate": float(y.mean()),
    },
    "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
    "interpretation_notes": (
        "GBM with k-fold OOF threshold tuning for F1 on imbalanced binary (~22.8% pos). "
        f"Small 3-config grid (per tune-when-n-supports-it skill); selected {final_label} "
        f"with thr={final_thr:.3f}. Other library skills' preconditions did not hold "
        "(no NaNs, no group IDs, no aux files, no ordinal target, single metric)."
    ),
    "elapsed_seconds": time.time() - t0,
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"[done] total_elapsed={time.time()-t0:.1f}s")
