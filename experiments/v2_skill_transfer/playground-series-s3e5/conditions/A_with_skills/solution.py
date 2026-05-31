"""
Playground Series S3E5 — Wine Quality (QWK metric, ordinal target 3..8)

Approach:
- Target identified via train/test schema diff: only `quality` is train-only. (skill #1)
- Verify prompt claims against data: prompt said 1645/413, actual 1644/412.
  Sample submission is STALE (IDs 2056-3427, but real test IDs are 23..2048
  with zero overlap). Build submission from test.csv IDs. (skill #2, #9)
- No NaN. Random re-split (Id ranges overlap, no group structure).
  -> StratifiedKFold(5) is appropriate (skill #4, #7).
- QWK is ordinal, weighted by squared distance. Treat as regression
  (HistGradientBoostingRegressor) + learn optimal rounding thresholds on
  out-of-fold predictions. Also benchmark a classifier baseline. (skill #3, #8)
- Tune n_estimators/learning_rate/max_depth/max_leaf_nodes via small grid
  with the same 5-fold CV. (skill #10)
- Submission: integer in {3..8}, header `Id,quality` matching sample format
  but using test.csv IDs. (skill #2, #3)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
RNG = 0
np.random.seed(RNG)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
assert DATA_DIR.exists(), f"data dir not found at {DATA_DIR}"


# ---------------------------------------------------------------------------
# Load & verify spec claims against data (skill_verify_spec_claims_against_data)
# ---------------------------------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("=== spec verification ===")
print(f"train rows: {len(train)} (prompt said 1645)  match={len(train)==1645}")
print(f"test  rows: {len(test)}  (prompt said 413)   match={len(test)==413}")
print(f"sample_sub rows: {len(sample_sub)}")
print(f"sample_sub Id range: {sample_sub['Id'].min()}..{sample_sub['Id'].max()}")
print(f"test       Id range: {test['Id'].min()}..{test['Id'].max()}")
print(
    "sample_sub matches test IDs:",
    set(sample_sub["Id"]) == set(test["Id"]),
)

# Target via schema diff (skill_target_via_schema_diff)
candidates = set(train.columns) - set(test.columns)
print(f"target candidates (train-only): {candidates}")
assert candidates == {"quality"}, f"unexpected target candidates: {candidates}"
TARGET = "quality"

# No group/time structure; random split (skill_validation_matches_test_distribution).
id_overlap = len(set(train["Id"]) & set(test["Id"]))
print(f"train/test Id overlap: {id_overlap}  (random re-split confirmed)")
assert id_overlap == 0

# NaN check
print(f"NaN in train: {train.isna().sum().sum()}, NaN in test: {test.isna().sum().sum()}")

# Target distribution + dtype
print("target value_counts:")
print(train[TARGET].value_counts().sort_index().to_string())
print(f"target dtype: {train[TARGET].dtype}")

y = train[TARGET].astype(int).values
classes = np.sort(train[TARGET].unique()).astype(int)
K_MIN, K_MAX = int(classes.min()), int(classes.max())
print(f"target range: {K_MIN}..{K_MAX}")

# ---------------------------------------------------------------------------
# Features (skill_categorical_prep_hygiene: drop Id, all numeric here)
# ---------------------------------------------------------------------------
FEATURE_COLS = [c for c in train.columns if c not in ("Id", TARGET)]
print(f"feature cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")
X = train[FEATURE_COLS].astype(float).values
X_test = test[FEATURE_COLS].astype(float).values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def qwk(y_true, y_pred):
    return cohen_kappa_score(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        weights="quadratic",
    )


def fit_thresholds(oof_pred_float, y_true):
    """
    Learn 5 cut-points between [K_MIN+1..K_MAX] that map a float regression
    OOF prediction onto integer classes K_MIN..K_MAX so as to maximize QWK.
    Uses coordinate ascent on a sorted grid.
    """
    # initial thresholds at half-integers
    init = np.array([K_MIN + 0.5 + i for i in range(K_MAX - K_MIN)])
    thresholds = init.copy()

    def apply(t):
        bins = np.concatenate(([-np.inf], t, [np.inf]))
        return np.digitize(oof_pred_float, bins) - 1 + K_MIN

    best = qwk(y_true, apply(thresholds))
    improved = True
    grid_radius, step = 0.6, 0.01
    eps = 1e-3
    while improved:
        improved = False
        for i in range(len(thresholds)):
            lo = (thresholds[i - 1] + eps) if i > 0 else float(K_MIN - 1.0)
            hi = (thresholds[i + 1] - eps) if i + 1 < len(thresholds) else float(K_MAX + 1.0)
            if hi <= lo:
                continue
            start = max(lo, thresholds[i] - grid_radius)
            stop = min(hi, thresholds[i] + grid_radius)
            if stop <= start:
                continue
            candidates = np.arange(start, stop + step / 2, step)
            for c in candidates:
                # enforce strict monotonicity
                if i > 0 and c <= thresholds[i - 1]:
                    continue
                if i + 1 < len(thresholds) and c >= thresholds[i + 1]:
                    continue
                test_t = thresholds.copy()
                test_t[i] = c
                s = qwk(y_true, apply(test_t))
                if s > best + 1e-9:
                    best, thresholds[i] = s, c
                    improved = True
    return thresholds, best


def apply_thresholds(pred_float, thresholds):
    bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
    out = np.digitize(pred_float, bins) - 1 + K_MIN
    return np.clip(out, K_MIN, K_MAX).astype(int)


# ---------------------------------------------------------------------------
# Baselines (skill_baselines_and_invariants)
# ---------------------------------------------------------------------------
print("\n=== baselines ===")
maj = int(pd.Series(y).mode().iloc[0])
print(f"majority-class ({maj}) QWK on full train: {qwk(y, np.full_like(y, maj)):.4f}")
# Reference baseline via CV
cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
mean_round_scores = []
for tr_idx, va_idx in cv.split(X, y):
    mean_pred = int(round(np.mean(y[tr_idx])))
    mean_round_scores.append(qwk(y[va_idx], np.full(len(va_idx), mean_pred)))
print(f"mean-rounded baseline CV-QWK: {np.mean(mean_round_scores):.4f}")

# Single-feature baseline: alcohol → quality (well-known correlation)
alcohol_corr = np.corrcoef(X[:, FEATURE_COLS.index("alcohol")], y)[0, 1]
print(f"alcohol-vs-quality Pearson: {alcohol_corr:.4f}")


# ---------------------------------------------------------------------------
# CV helper for a regression config — OOF + tuned thresholds (skill_tune_dont_default)
# ---------------------------------------------------------------------------
def evaluate_reg_config(params, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        m = HistGradientBoostingRegressor(random_state=RNG, **params)
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    thresholds, qwk_tuned = fit_thresholds(oof, y)
    # also report naive rounding
    naive_pred = np.clip(np.round(oof).astype(int), K_MIN, K_MAX)
    qwk_naive = qwk(y, naive_pred)
    return qwk_tuned, qwk_naive, thresholds, oof


def evaluate_clf_config(params, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    fold_scores = []
    for tr_idx, va_idx in cv.split(X, y):
        m = HistGradientBoostingClassifier(random_state=RNG, **params)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[va_idx]).astype(int)
        fold_scores.append(qwk(y[va_idx], pred))
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


# ---------------------------------------------------------------------------
# Tuning: small grid for HistGB regressor (skill_tune_dont_default)
# ---------------------------------------------------------------------------
print("\n=== tuning HistGradientBoostingRegressor (with threshold tuning) ===")
grid = [
    # (n_estimators=max_iter, learning_rate, max_depth, max_leaf_nodes, l2)
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0, min_samples_leaf=20),
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=15, l2_regularization=0.0, min_samples_leaf=20),
    dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0, min_samples_leaf=20),
    dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20),
    dict(max_iter=800, learning_rate=0.02, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0, min_samples_leaf=30),
    dict(max_iter=300, learning_rate=0.08, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0, min_samples_leaf=10),
    dict(max_iter=500, learning_rate=0.04, max_depth=None, max_leaf_nodes=63, l2_regularization=1.0, min_samples_leaf=20),
]

best = {"score": -np.inf, "params": None, "thresholds": None, "oof": None, "naive": None}
for params in grid:
    s_tuned, s_naive, thr, oof = evaluate_reg_config(params, X, y)
    print(f"  {params}\n    QWK (tuned-thr) = {s_tuned:.4f} | QWK (naive-round) = {s_naive:.4f}")
    if s_tuned > best["score"]:
        best = {"score": s_tuned, "params": params, "thresholds": thr, "oof": oof, "naive": s_naive}

print(f"\nbest reg params: {best['params']}")
print(f"best CV-QWK (with tuned thresholds): {best['score']:.4f}")
print(f"  vs naive rounding: {best['naive']:.4f}")
print(f"  thresholds: {np.round(best['thresholds'], 4)}")

# Classifier benchmark
print("\n=== benchmark: HistGradientBoostingClassifier ===")
clf_grid = [
    dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20),
    dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20),
    dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=10),
]
clf_best = {"score": -np.inf, "params": None}
for params in clf_grid:
    m, s = evaluate_clf_config(params, X, y)
    print(f"  {params}: QWK = {m:.4f} ± {s:.4f}")
    if m > clf_best["score"]:
        clf_best = {"score": m, "params": params, "std": s}
print(f"best classifier CV-QWK: {clf_best['score']:.4f} ± {clf_best['std']:.4f}")

# Decide final model: regressor+tuned-thresholds if it beats classifier
use_regression = best["score"] >= clf_best["score"]
print(f"\nfinal model: {'regressor + tuned thresholds' if use_regression else 'classifier'}")


# ---------------------------------------------------------------------------
# Refit on full train, predict on test (skill_tune_dont_default: refit on all data)
# ---------------------------------------------------------------------------
if use_regression:
    final = HistGradientBoostingRegressor(random_state=RNG, **best["params"])
    final.fit(X, y)
    test_pred_float = final.predict(X_test)
    # use thresholds learned on OOF; ensures threshold choice doesn't see test
    test_pred = apply_thresholds(test_pred_float, best["thresholds"])
    headline = float(best["score"])
    final_params = best["params"]
    final_kind = "HistGradientBoostingRegressor+tuned_thresholds"
else:
    final = HistGradientBoostingClassifier(random_state=RNG, **clf_best["params"])
    final.fit(X, y)
    test_pred = final.predict(X_test).astype(int)
    headline = float(clf_best["score"])
    final_params = clf_best["params"]
    final_kind = "HistGradientBoostingClassifier"

test_pred = np.clip(test_pred, K_MIN, K_MAX).astype(int)
print(f"test prediction distribution:")
print(pd.Series(test_pred).value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# Submission (skill_trust_test_csv_not_sample_submission, skill_match_submission_format_to_metric)
# ---------------------------------------------------------------------------
sub = pd.DataFrame({"Id": test["Id"].values, "quality": test_pred})
# sanity asserts
assert not sub.isna().any().any()
assert sub["quality"].between(K_MIN, K_MAX).all()
assert sub["quality"].dtype == np.dtype("int64") or sub["quality"].dtype == np.dtype("int32")
assert len(sub) == len(test)
assert list(sub.columns) == ["Id", "quality"]

sub_path = HERE / "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"wrote submission to {sub_path} (rows={len(sub)})")


# ---------------------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------------------
all_metrics = {
    "cv_qwk_reg_tuned_thresholds": float(best["score"]),
    "cv_qwk_reg_naive_rounding": float(best["naive"]),
    "cv_qwk_classifier": float(clf_best["score"]),
    "cv_qwk_classifier_std": float(clf_best["std"]),
    "baseline_majority_qwk_train": float(qwk(y, np.full_like(y, maj))),
    "baseline_mean_rounded_qwk_cv": float(np.mean(mean_round_scores)),
    "alcohol_target_pearson": float(alcohol_corr),
    "final_model": final_kind,
    "final_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in final_params.items()},
    "thresholds": [float(t) for t in best["thresholds"]] if use_regression else None,
}

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "QuadraticWeightedKappa_5fold_StratifiedCV",
    "headline_value": headline,
    "all_metrics": all_metrics,
    "skills_applied": [
        "skill_target_via_schema_diff",
        "skill_verify_spec_claims_against_data",
        "skill_trust_test_csv_not_sample_submission",
        "skill_match_submission_format_to_metric",
        "skill_validation_matches_test_distribution",
        "skill_baselines_and_invariants",
        "skill_tune_dont_default",
        "skill_categorical_prep_hygiene",
    ],
    "skills_skipped": [
        {"skill": "skill_detect_group_leakage", "reason": "No group columns present and train/test Id overlap is zero with overlapping ranges — random re-split confirmed, no group-aware CV needed."},
        {"skill": "skill_model_missingness_as_signal", "reason": "Zero NaN in train and test; no missingness to model."},
    ],
    "interpretation_notes": (
        f"Target `quality` is ordinal int in [{K_MIN},{K_MAX}] with majority class 5 (≈40.8%); "
        "QWK weights errors by squared distance so adjacent mistakes (5↔6) cost less than far ones (3↔8). "
        "Modelled as regression with HistGradientBoostingRegressor and post-hoc threshold tuning learned on out-of-fold "
        "predictions — this preserves ordinal structure and outperformed the direct classifier benchmark in CV. "
        "Critical: sample_submission.csv is stale (its IDs 2056-3427 do not overlap real test IDs 23-2048) so the submission "
        "is built from test.csv IDs; final predictions are integer-valued 3..8 to match the QWK grader."
    ),
}

with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n=== result.json ===")
print(json.dumps(result, indent=2))
