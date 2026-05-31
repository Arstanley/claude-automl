"""
Disaster-tweet binary classification (Kaggle NLP getting started).

Pipeline:
- Text feature = tweet text + ' [KW] ' + keyword token (NA -> '__missing__')
  combined with location_missing indicator feature.
- TF-IDF (word 1-2 + char 3-5) -> LogisticRegression.
- 5-fold stratified CV: tune C, generate OOF probs, tune decision threshold for F1
  (skill_tune_decision_threshold_for_f1).
- Refit on full train, predict on test, apply tuned threshold.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/nlp-getting-started/data"
)
WORK = Path(__file__).resolve().parent
N_JOBS = 2
SEED = 0

t0 = time.time()
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# --- skill_missing_indicator_when_missingness_signals_class -------------------
# location has 33% NaN in train; keyword has <1% — add indicator for location.
loc_miss_train = train["location"].isna().astype(int).values
loc_miss_test = test["location"].isna().astype(int).values
kw_miss_train = train["keyword"].isna().astype(int).values
kw_miss_test = test["keyword"].isna().astype(int).values
indicators_added = ["location"]
if train["keyword"].isna().mean() >= 0.005:
    indicators_added.append("keyword")

# Build text feature: tweet text + keyword (urlencoded -> underscore decoded)
def build_text(df: pd.DataFrame) -> pd.Series:
    kw = df["keyword"].fillna("").str.replace("%20", " ", regex=False)
    text = df["text"].fillna("")
    return (text + " __KW__ " + kw).astype(str)

X_text_train = build_text(train)
X_text_test = build_text(test)
y = train["target"].values.astype(int)

# --- TF-IDF features (word + char) -------------------------------------------
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

Xw_train = word_vec.fit_transform(X_text_train)
Xw_test = word_vec.transform(X_text_test)
Xc_train = char_vec.fit_transform(X_text_train)
Xc_test = char_vec.transform(X_text_test)

ind_train = csr_matrix(np.column_stack([loc_miss_train, kw_miss_train]).astype(float))
ind_test = csr_matrix(np.column_stack([loc_miss_test, kw_miss_test]).astype(float))

X = hstack([Xw_train, Xc_train, ind_train]).tocsr()
X_test = hstack([Xw_test, Xc_test, ind_test]).tocsr()
print(f"feature matrix shape: train={X.shape}, test={X_test.shape}")

# --- CV: tune C, get OOF probs ----------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
C_GRID = [0.5, 1.0, 2.0, 4.0]
oof_by_C: dict[float, np.ndarray] = {}
auc_by_C: dict[float, float] = {}

for C in C_GRID:
    oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X, y):
        clf = LogisticRegression(
            C=C, solver="liblinear", max_iter=1000, random_state=SEED
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    oof_by_C[C] = oof
    # Score at 0.5 default for picking C (rough — final F1 uses tuned threshold)
    f1_at_default = f1_score(y, (oof >= 0.5).astype(int))
    auc_by_C[C] = f1_at_default
    print(f"  C={C}: OOF F1@0.5={f1_at_default:.4f}")

best_C = max(auc_by_C, key=auc_by_C.get)
oof = oof_by_C[best_C]
print(f"chosen C={best_C}")

# --- skill_tune_decision_threshold_for_f1 ------------------------------------
grid = np.linspace(0.20, 0.80, 61)
scores = [f1_score(y, (oof >= t).astype(int)) for t in grid]
best_idx = int(np.argmax(scores))
best_thr = float(grid[best_idx])
best_f1 = float(scores[best_idx])
f1_at_default = f1_score(y, (oof >= 0.5).astype(int))
p_at_default = precision_score(y, (oof >= 0.5).astype(int))
r_at_default = recall_score(y, (oof >= 0.5).astype(int))
p_at_thr = precision_score(y, (oof >= best_thr).astype(int))
r_at_thr = recall_score(y, (oof >= best_thr).astype(int))
print(
    f"threshold tuning: 0.5 -> F1={f1_at_default:.4f} (P={p_at_default:.3f} R={r_at_default:.3f}); "
    f"thr={best_thr:.3f} -> F1={best_f1:.4f} (P={p_at_thr:.3f} R={r_at_thr:.3f})"
)

# Print a few around the optimum for auditability
for i in range(max(0, best_idx - 2), min(len(grid), best_idx + 3)):
    print(f"  thr={grid[i]:.3f}  F1={scores[i]:.4f}")

# --- Refit on full data, predict on test, apply tuned threshold -------------
final = LogisticRegression(C=best_C, solver="liblinear", max_iter=1000, random_state=SEED)
final.fit(X, y)
test_proba = final.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)

# --- submission.csv ----------------------------------------------------------
submission = pd.DataFrame({"id": test["id"].values, "target": test_pred})
submission.to_csv(WORK / "submission.csv", index=False)

# --- result.json -------------------------------------------------------------
skills_applied = [
    "skill_tune_decision_threshold_for_f1.md",
    "skill_missing_indicator_when_missingness_signals_class.md",
]
skills_evaluated_not_applied = [
    "skill_probe_group_structure_in_ids.md: no >=5% train/test overlap on non-target cols; ids are integers without compound structure.",
    "skill_train_on_listed_auxiliary_files.md: prompt lists only train/test/sample_submission; no aux file.",
    "skill_verify_split_claim_before_holdout.md: prompt makes no structural train/test split claim.",
    "skill_stringify_low_cardinality_int_codes.md: no low-cardinality int-coded features in scope (text task).",
    "skill_multi_axis_constraint_budgets.md: only F1 mentioned; no latency/size constraint axes.",
    "skill_tune_when_n_train_supports_it.md: model family is LogisticRegression on TF-IDF (not GBM/RF). C-grid still done.",
    "skill_log_target_for_skewed_nonnegative_regression.md: classification task, not regression.",
    "skill_blend_two_model_families.md: task is pure-text (TF-IDF features), skill precondition restricts to tabular.",
]

result = {
    "target_column_chosen": "target",
    "headline_metric": "f1",
    "headline_value": best_f1,
    "all_metrics": {
        "oof_f1_at_default_threshold": float(f1_at_default),
        "oof_f1_at_tuned_threshold": float(best_f1),
        "oof_precision_at_default": float(p_at_default),
        "oof_recall_at_default": float(r_at_default),
        "oof_precision_at_tuned": float(p_at_thr),
        "oof_recall_at_tuned": float(r_at_thr),
    },
    "skills_applied": skills_applied,
    "skills_evaluated_not_applied": skills_evaluated_not_applied,
    "interpretation_notes": (
        f"TF-IDF (word 1-2 + char 3-5) + LogisticRegression on text+keyword, "
        f"with location/keyword missing indicators. 5-fold stratified CV, C={best_C}. "
        f"Decision threshold tuned to {best_thr:.3f} on OOF probs "
        f"(F1 {f1_at_default:.4f} -> {best_f1:.4f}; recall up from {r_at_default:.3f} to {r_at_thr:.3f})."
    ),
    "audit": {
        "n_train": int(len(y)),
        "n_test": int(len(test)),
        "class_balance_positive": float(y.mean()),
        "chosen_C": float(best_C),
        "chosen_threshold": float(best_thr),
        "chosen_threshold_metric_value": float(best_f1),
        "threshold_grid": [float(x) for x in grid.tolist()],
        "C_grid_oof_f1_at_default": {str(C): float(v) for C, v in auc_by_C.items()},
        "missingness_indicators_added": indicators_added,
        "feature_matrix_shape_train": list(X.shape),
        "feature_matrix_shape_test": list(X_test.shape),
        "runtime_seconds": round(time.time() - t0, 2),
    },
}

with open(WORK / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nDONE in {time.time() - t0:.1f}s. headline F1 (OOF, tuned threshold) = {best_f1:.4f}")
print(f"  submission.csv rows: {len(submission)}; result.json written.")
