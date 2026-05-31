"""
Steel Plates Faults - playground-series-s4e3
Multi-label classification (7 binary targets); metric = mean AUC.
Approach: per-target HistGradientBoostingClassifier with a small tuned config
selected via 5-fold OOF on each target. n_train=15375 supports tuning.
Skills applied:
  - skill_tune_when_n_train_supports_it (n>=500, GBM family, no size/latency)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e3/data")
OUT_DIR = Path(__file__).parent

TARGETS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    feat_cols = [c for c in train.columns if c not in TARGETS + ["id"]]
    X = train[feat_cols].values
    X_test = test[feat_cols].values

    # Small explicit grid — enough variance to beat defaults, small enough to fit budget
    grid = [
        dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0),
        dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
    ]

    rng_seed = 42
    n_splits = 5
    oof = np.zeros((len(train), len(TARGETS)))
    test_pred = np.zeros((len(test), len(TARGETS)))
    aucs = {}
    chosen = {}

    for ti, tgt in enumerate(TARGETS):
        y = train[tgt].values.astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)

        best_cfg = None
        best_oof = None
        best_test = None
        best_auc = -np.inf

        for cfg in grid:
            oof_p = np.zeros(len(train))
            test_p = np.zeros(len(test))
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
                clf = HistGradientBoostingClassifier(
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=rng_seed + fold,
                    **cfg,
                )
                clf.fit(X[tr_idx], y[tr_idx])
                oof_p[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
                test_p += clf.predict_proba(X_test)[:, 1] / n_splits
            auc = roc_auc_score(y, oof_p)
            if auc > best_auc:
                best_auc = auc
                best_cfg = cfg
                best_oof = oof_p
                best_test = test_p

        oof[:, ti] = best_oof
        test_pred[:, ti] = best_test
        aucs[tgt] = float(best_auc)
        chosen[tgt] = best_cfg
        print(f"[{tgt}] AUC={best_auc:.4f} cfg={best_cfg}")

    mean_auc = float(np.mean(list(aucs.values())))
    print(f"\nMean AUC: {mean_auc:.4f}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    # Submission
    sub = pd.DataFrame({"id": test["id"].values})
    for ti, tgt in enumerate(TARGETS):
        sub[tgt] = test_pred[:, ti]
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # Result JSON
    result = {
        "target_column_chosen": TARGETS,
        "headline_metric": "mean_auc",
        "headline_value": mean_auc,
        "all_metrics": {
            "per_target_auc": aucs,
            "mean_auc": mean_auc,
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "Multi-label binary classification with 7 targets. Trained one "
            "HistGradientBoostingClassifier per target with 5-fold stratified CV; "
            "selected from a 3-config grid (learning_rate/max_iter/leaves/regularization) "
            "by OOF AUC. n_train=15375, no missing values, no size/latency constraints."
        ),
        "chosen_configs": chosen,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
