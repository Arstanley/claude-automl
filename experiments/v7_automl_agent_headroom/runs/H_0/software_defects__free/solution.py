"""Software Defects binary classification.

Metric: F1 (binary). Target: `defects` (True/False -> 1/0).
Approach:
  - HistGradientBoostingClassifier (sklearn) with a small explicit config grid.
  - Stratified train/val split, pick best config by val F1 (with threshold tuning).
  - Tune decision threshold on val out-of-fold probabilities for max F1.
  - Refit best config on full train, predict test using tuned threshold.

Skills applied:
  - skill_tune_when_n_train_supports_it: n_train=81410 with GBM family, no size/latency constraints.
  - skill_probe_group_structure_in_ids: probed id overlap (none) and ranges (interleaved) -> random stratified split is safe.
"""

import os
import json
import time
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/software_defects/data"
N_JOBS = 2
RNG = 42

t0 = time.time()

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

y = train["defects"].astype(int).values
feature_cols = [c for c in train.columns if c not in ("id", "defects")]
X = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
test_ids = test["id"].values

print(f"train={X.shape}, test={X_test.shape}, pos_rate={y.mean():.4f}")


def tune_threshold(y_true, y_proba):
    """Find threshold maximizing F1."""
    # sweep a reasonable grid
    ths = np.linspace(0.05, 0.9, 86)
    best_t, best_f = 0.5, -1.0
    for t in ths:
        f = f1_score(y_true, (y_proba >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


# Hold out a stratified validation set for threshold + config selection.
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=y
)

# Small explicit grid: vary depth/leaves/lr/regularization.
configs = [
    dict(max_depth=None, max_leaf_nodes=31, learning_rate=0.07, max_iter=500, l2_regularization=0.0, min_samples_leaf=20),
    dict(max_depth=None, max_leaf_nodes=63, learning_rate=0.05, max_iter=600, l2_regularization=0.0, min_samples_leaf=20),
    dict(max_depth=None, max_leaf_nodes=31, learning_rate=0.05, max_iter=700, l2_regularization=1.0, min_samples_leaf=30),
    dict(max_depth=None, max_leaf_nodes=127, learning_rate=0.05, max_iter=500, l2_regularization=0.0, min_samples_leaf=40),
]

results = []
best = None
for i, cfg in enumerate(configs):
    if time.time() - t0 > 140:
        print(f"time budget: skipping config {i}")
        break
    clf = HistGradientBoostingClassifier(
        random_state=RNG,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        **cfg,
    )
    clf.fit(X_tr, y_tr)
    proba_val = clf.predict_proba(X_val)[:, 1]
    th, f1_at_th = tune_threshold(y_val, proba_val)
    f1_at_05 = f1_score(y_val, (proba_val >= 0.5).astype(int))
    results.append({
        "config": cfg,
        "val_f1_at_0.5": f1_at_05,
        "val_f1_tuned": f1_at_th,
        "threshold": th,
        "n_iter": int(getattr(clf, "n_iter_", -1)),
        "elapsed": time.time() - t0,
    })
    print(f"cfg{i}: f1@0.5={f1_at_05:.4f}, f1_tuned={f1_at_th:.4f}@t={th:.3f}, n_iter={clf.n_iter_}, t={time.time()-t0:.1f}s")
    if best is None or f1_at_th > best["val_f1_tuned"]:
        best = {
            "cfg_idx": i,
            "cfg": cfg,
            "val_f1_tuned": f1_at_th,
            "threshold": th,
            "val_f1_at_0.5": f1_at_05,
        }

print(f"best: {best}")

# Refit best config on full train (for max signal), keep the tuned threshold.
final = HistGradientBoostingClassifier(
    random_state=RNG,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
    **best["cfg"],
)
final.fit(X, y)
test_proba = final.predict_proba(X_test)[:, 1]

# Sanity: also report val performance of refit-on-tr-only config (already have).
threshold = best["threshold"]
test_pred = (test_proba >= threshold).astype(int)
print(f"test positive rate: {test_pred.mean():.4f} (train pos rate: {y.mean():.4f})")

sub = pd.DataFrame({"id": test_ids, "defects": test_pred})
sub.to_csv("submission.csv", index=False)

result = {
    "target_column_chosen": "defects",
    "headline_metric": "f1",
    "headline_value": best["val_f1_tuned"],
    "all_metrics": {
        "val_f1": best["val_f1_tuned"],
        "val_f1_at_0.5": best["val_f1_at_0.5"],
        "threshold": threshold,
        "configs_tried": results,
        "best_cfg_idx": best["cfg_idx"],
    },
    "skills_applied": [
        "skill_tune_when_n_train_supports_it.md",
        "skill_probe_group_structure_in_ids.md",
    ],
    "interpretation_notes": (
        "HistGradientBoostingClassifier with a 4-config grid (varying leaves/lr/l2). "
        "Selected the config maximizing val F1 after threshold tuning on a 20% stratified holdout; "
        "refit on full train and applied tuned threshold to test. F1 is sensitive to the threshold under "
        "class imbalance (~23% positive)."
    ),
    "elapsed_sec": time.time() - t0,
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"done in {time.time()-t0:.1f}s, val_f1={best['val_f1_tuned']:.4f}")
