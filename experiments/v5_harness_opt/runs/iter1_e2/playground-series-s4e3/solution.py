"""
Solution for playground-series-s4e3: multi-label binary classification of
7 steel-plate defect categories. Metric = mean column-wise ROC AUC.

Approach:
- Per-target HistGradientBoostingClassifier (tree-based, no scaling needed).
- Small explicit hyperparameter grid (3 configs) selected per target by 3-fold CV.
- n_jobs/threads capped via OMP_NUM_THREADS=2 (set externally), models run sequentially.

Skills applied:
- skill_tune_when_n_train_supports_it (n_train=15375 >> 500; no constraint tokens)
- (skipped) skill_stringify_low_cardinality_int_codes — tree-only pipeline, no scaler
- (skipped) skill_class_weight_for_imbalanced_binary — AUC is rank-only and HGB does
  not accept class_weight; sample_weight tweaks tend to be negligible for AUC.
"""
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

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e3/data")
OUT_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/iter1_e2/playground-series-s4e3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

GRID = [
    dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0),
    dict(max_iter=500, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=1.0),
    dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=0.0),
]


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"train={train.shape} test={test.shape}")

    feat_cols = [c for c in train.columns if c not in TARGETS + ["id"]]
    X = train[feat_cols].to_numpy(dtype=np.float32)
    X_test = test[feat_cols].to_numpy(dtype=np.float32)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    per_target_auc = {}
    chosen_configs = {}
    test_preds = pd.DataFrame({"id": test["id"]})
    grid_log = {}

    for target in TARGETS:
        y = train[target].to_numpy()
        config_scores = []
        for params in GRID:
            fold_aucs = []
            for tr_idx, va_idx in cv.split(X, y):
                m = HistGradientBoostingClassifier(
                    **params, random_state=0, early_stopping=False
                )
                m.fit(X[tr_idx], y[tr_idx])
                p = m.predict_proba(X[va_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y[va_idx], p))
            config_scores.append((float(np.mean(fold_aucs)), params))
        config_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_params = config_scores[0]
        per_target_auc[target] = best_score
        chosen_configs[target] = best_params
        grid_log[target] = [(s, p) for s, p in config_scores]

        # Refit on all train, predict on test
        final = HistGradientBoostingClassifier(
            **best_params, random_state=0, early_stopping=False
        )
        final.fit(X, y)
        test_preds[target] = final.predict_proba(X_test)[:, 1]
        print(f"{target}: best AUC={best_score:.4f} params={best_params}")

    mean_auc = float(np.mean(list(per_target_auc.values())))
    print(f"\nMean CV AUC across 7 targets: {mean_auc:.4f}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Write submission
    sub = test_preds[["id"] + TARGETS]
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"Wrote {OUT_DIR/'submission.csv'} ({len(sub)} rows)")

    # Write result.json
    result = {
        "target_column_chosen": TARGETS,
        "headline_metric": "mean_column_roc_auc",
        "headline_value": mean_auc,
        "all_metrics": {
            "mean_column_roc_auc": mean_auc,
            "per_target_roc_auc": per_target_auc,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Multi-label binary (7 defect classes); per-target HistGradientBoosting "
            "with a 3-config grid selected by 3-fold stratified CV. Tree model so no "
            "scaling/stringification needed; AUC is rank-only so class_weight skipped."
        ),
        "audit": {
            "hyperparameter_grid": GRID,
            "chosen_config_per_target": chosen_configs,
            "grid_cv_scores": {t: [{"mean_auc": s, "params": p} for s, p in lst]
                               for t, lst in grid_log.items()},
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": len(feat_cols),
            "cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=0)",
        },
    }
    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_DIR/'result.json'}")


if __name__ == "__main__":
    main()
