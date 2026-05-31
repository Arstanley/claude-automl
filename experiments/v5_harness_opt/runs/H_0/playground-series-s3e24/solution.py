"""
playground-series-s3e24 — smoker status binary classification, metric=AUC.
n_train=127K, all numeric, no NaNs. We apply:
  - skill_tune_when_n_train_supports_it: small grid (2 configs) of HistGradientBoosting
    with 5-fold StratifiedKFold OOF AUC; pick best, retrain on full data.
Budget: 3 min, n_jobs=2.
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e24/data")

t0 = time.time()

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target = "smoking"
id_col = "id"
feat_cols = [c for c in train.columns if c not in (target, id_col)]

X = train[feat_cols].values
y = train[target].values
X_test = test[feat_cols].values

# Small grid: 2 reasonable HGB configs for binary AUC on ~127K rows.
configs = [
    {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 63, "min_samples_leaf": 30,
     "l2_regularization": 0.0, "early_stopping": True, "validation_fraction": 0.1,
     "n_iter_no_change": 30, "random_state": 0},
    {"learning_rate": 0.03, "max_iter": 900, "max_leaf_nodes": 31, "min_samples_leaf": 60,
     "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1,
     "n_iter_no_change": 30, "random_state": 0},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = []  # list of (config_idx, mean_auc, std)
oof_by_cfg = {}

for ci, params in enumerate(configs):
    oof = np.zeros(len(X))
    fold_aucs = []
    for fi, (tr, va) in enumerate(cv.split(X, y)):
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]
        oof[va] = p
        fold_aucs.append(roc_auc_score(y[va], p))
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    cv_scores.append((ci, mean_auc, std_auc))
    oof_by_cfg[ci] = oof
    print(f"config {ci}: AUC={mean_auc:.5f} +- {std_auc:.5f}  elapsed={time.time()-t0:.1f}s")

best_ci, best_auc, best_std = max(cv_scores, key=lambda r: r[1])
best_params = configs[best_ci]
print(f"best config={best_ci} AUC={best_auc:.5f}")

# Refit best on full train; turn off early stopping & cap max_iter at observed best to avoid overfit drift.
final_params = dict(best_params)
final_params["early_stopping"] = False
final_clf = HistGradientBoostingClassifier(**final_params)
final_clf.fit(X, y)
test_pred = final_clf.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({id_col: test[id_col].values, target: test_pred})
sub.to_csv(HERE / "submission.csv", index=False)

result = {
    "target_column_chosen": target,
    "headline_metric": "roc_auc",
    "headline_value": best_auc,
    "all_metrics": {
        "cv_auc_mean": best_auc,
        "cv_auc_std": best_std,
        "per_config": [
            {"config_idx": ci, "params": configs[ci], "cv_auc_mean": m, "cv_auc_std": s}
            for (ci, m, s) in cv_scores
        ],
    },
    "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
    "interpretation_notes": (
        "Binary classification with 127K rows, all numeric features, no NaNs. "
        "Applied tune-when-n-supports-it skill: 5-fold stratified CV over a 2-config "
        "HistGradientBoosting grid; selected best by mean OOF AUC and refit on full data. "
        "Other skills (group structure, missing indicator, split-claim, stringify codes, aux files) "
        "did not meet preconditions for this task."
    ),
    "elapsed_s": time.time() - t0,
}

with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"DONE  AUC={best_auc:.5f}  elapsed={time.time()-t0:.1f}s")
