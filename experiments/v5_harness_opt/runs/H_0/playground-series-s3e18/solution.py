"""
Solution for Kaggle Playground Series S3E18: Multi-label EC1/EC2 prediction.

Task: Binary classification (multi-output) — predict probabilities for EC1 and EC2.
Metric: Mean of column-wise ROC AUC.
Train: 11,870 rows x 33 features (molecular descriptors). No missing values.
Test: 2,968 rows. No id overlap with train.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=11870 >> 500, GBM family,
  no size/latency tokens in prompt -> tune with a small 3-config grid.

Approach:
- One HistGradientBoostingClassifier per target (EC1, EC2). HGB is fast, handles
  feature scaling implicitly, and respects an n_jobs-equivalent via threading.
- 5-fold stratified CV per target across 3 configs; pick best mean ROC AUC per target.
- Refit on full train, predict test probabilities.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e18/data"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

TARGETS = ["EC1", "EC2"]
ALL_TRAIN_TARGETS = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]

GRID = [
    dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0),
    dict(max_iter=500, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=1.0),
    dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=0.0),
]


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    feature_cols = [c for c in train.columns if c not in (["id"] + ALL_TRAIN_TARGETS)]
    X = train[feature_cols].to_numpy()
    X_test = test[feature_cols].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    per_target_metrics = {}
    test_pred = pd.DataFrame({"id": test["id"]})
    aucs = []

    for target in TARGETS:
        y = train[target].to_numpy()
        config_scores = []
        for cfg in GRID:
            fold_aucs = []
            for tr_idx, va_idx in cv.split(X, y):
                model = HistGradientBoostingClassifier(random_state=0, **cfg)
                model.fit(X[tr_idx], y[tr_idx])
                p = model.predict_proba(X[va_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y[va_idx], p))
            mean_auc = float(np.mean(fold_aucs))
            std_auc = float(np.std(fold_aucs))
            config_scores.append((mean_auc, std_auc, cfg))
            print(f"[{target}] {cfg} -> AUC {mean_auc:.4f} +/- {std_auc:.4f}")

        config_scores.sort(key=lambda r: r[0], reverse=True)
        best_mean, best_std, best_cfg = config_scores[0]
        print(f"[{target}] BEST: {best_cfg} mean_auc={best_mean:.4f}")

        # Refit on full train
        final = HistGradientBoostingClassifier(random_state=0, **best_cfg)
        final.fit(X, train[target].to_numpy())
        test_pred[target] = final.predict_proba(X_test)[:, 1]

        per_target_metrics[target] = {
            "cv_mean_auc": best_mean,
            "cv_std_auc": best_std,
            "chosen_config": best_cfg,
            "grid_scores": [(m, s, c) for m, s, c in config_scores],
        }
        aucs.append(best_mean)

    headline = float(np.mean(aucs))
    print(f"Mean CV AUC (EC1, EC2): {headline:.4f}")

    # Write submission
    sub_path = os.path.join(WORK_DIR, "submission.csv")
    test_pred[["id", "EC1", "EC2"]].to_csv(sub_path, index=False)
    print(f"Wrote {sub_path} with shape {test_pred.shape}")

    result = {
        "target_column_chosen": "EC1,EC2 (multi-output)",
        "headline_metric": "mean_column_roc_auc",
        "headline_value": headline,
        "all_metrics": {
            "EC1_cv_auc": per_target_metrics["EC1"]["cv_mean_auc"],
            "EC2_cv_auc": per_target_metrics["EC2"]["cv_mean_auc"],
            "mean_cv_auc": headline,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "n_train=11870 >> 500 and no size/latency constraint in prompt, so the "
            "tune-when-large skill applies; ran a 3-config HistGradientBoostingClassifier "
            "grid with 5-fold stratified CV per target (EC1, EC2) and refit best config on "
            "full train. HGB handles raw molecular descriptors directly (no scaling) and "
            "stays well under the 2-minute budget. EC3-EC6 columns ignored (not scored)."
        ),
        "audit": {
            "hyperparameter_grid": GRID,
            "per_target": per_target_metrics,
            "wallclock_seconds": time.time() - t0,
        },
    }

    result_path = os.path.join(WORK_DIR, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {result_path}")
    print(f"Wall clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
