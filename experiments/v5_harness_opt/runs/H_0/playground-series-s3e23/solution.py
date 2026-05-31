"""
Playground Series S3E23 — Software Defects (binary classification, ROC AUC).

Approach:
- 21 numeric features, no missing values, n=81,410, ~23% positive.
- Apply skill_tune_when_n_train_supports_it: small explicit 2-config grid with
  HistGradientBoostingClassifier (handles non-linear interactions well on tabular
  numeric data) selected by 3-fold stratified CV AUC. n_jobs=2, time budget ~3 min.
- Skipped: probe_group_structure (sequential int IDs, no overlap),
  missing_indicator (zero NaNs), stringify_low_card_int_codes (no integer codes
  with low cardinality + linear/distance model — using GBM).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e23/data")

TARGET = "defects"
ID = "id"


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train={train.shape} test={test.shape}", flush=True)

    y = train[TARGET].astype(int).values
    feats = [c for c in train.columns if c not in (TARGET, ID)]
    X = train[feats].values
    X_test = test[feats].values

    # Small 2-config grid (within 3-min budget): vary learning_rate / leaves trade-off.
    # Early stopping caps actual iters well below max_iter.
    grid = [
        {"learning_rate": 0.06, "max_leaf_nodes": 31, "max_iter": 300,
         "min_samples_leaf": 30, "l2_regularization": 0.0},
        {"learning_rate": 0.08, "max_leaf_nodes": 63, "max_iter": 300,
         "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []
    for i, params in enumerate(grid):
        clf = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv, n_jobs=2)
        mean, std = float(scores.mean()), float(scores.std())
        print(f"cfg{i} {params}: AUC={mean:.5f} +/- {std:.5f}  (t={time.time()-t0:.1f}s)",
              flush=True)
        results.append({"params": params, "cv_auc_mean": mean, "cv_auc_std": std,
                        "fold_aucs": [float(s) for s in scores]})

    best_idx = int(np.argmax([r["cv_auc_mean"] for r in results]))
    best = results[best_idx]
    print(f"best=cfg{best_idx}: AUC={best['cv_auc_mean']:.5f}", flush=True)

    # Refit best on full train and predict on test.
    final = HistGradientBoostingClassifier(
        **best["params"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    final.fit(X, y)
    proba = final.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({ID: test[ID].values, TARGET: proba})
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"wrote submission.csv shape={sub.shape}", flush=True)

    out = {
        "target_column_chosen": TARGET,
        "headline_metric": "roc_auc_3fold_cv",
        "headline_value": best["cv_auc_mean"],
        "all_metrics": {
            "cv_auc_mean": best["cv_auc_mean"],
            "cv_auc_std": best["cv_auc_std"],
            "fold_aucs": best["fold_aucs"],
            "best_config_index": best_idx,
            "all_configs": results,
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "21 numeric features, no NaNs, n=81K, ~23% positive. Applied "
            "tune_when_n_train_supports_it: 2-config HistGradientBoosting grid "
            "selected by 3-fold stratified CV AUC, then refit on full train. "
            "Skipped probe-group/missing-indicator/stringify skills as their "
            "preconditions did not hold."
        ),
    }
    (HERE / "result.json").write_text(json.dumps(out, indent=2))
    print(f"wrote result.json; total t={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
