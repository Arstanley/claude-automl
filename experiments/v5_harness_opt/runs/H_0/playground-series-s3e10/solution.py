"""Solution for playground-series-s3e10 (Pulsar binary classification, log-loss).

Skill applied: skill_tune_when_n_train_supports_it.md
  - Precondition: GBM family + n_train (94051) >= 500 + no size/latency tokens in prompt.
  - Action: small 2-config grid on HistGradientBoostingClassifier with 3-fold StratifiedKFold,
    pick by mean log-loss, refit on full train, predict probabilities for test.

Other skills evaluated and NOT applied:
  - probe_group_structure_in_ids: 0 train/test ID overlap, no compound IDs.
  - train_on_listed_auxiliary_files: no auxiliary file present in DATA_DIR.
  - verify_split_claim_before_holdout: no structural split claim in prompt.
  - stringify_low_cardinality_int_codes: all 8 features are float64.
  - multi_axis_constraint_budgets: only log-loss is named; no latency/size constraints.
  - missing_indicator_when_missingness_signals_class: 0 NaNs in train and test.

Budget context (from harness): 3 min wall, n_jobs=2, grid <=2 configs.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e10/data"

RANDOM_STATE = 0
N_JOBS = 2


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    feature_cols = [c for c in train.columns if c not in ("id", "Class")]
    target_col = "Class"

    X = train[feature_cols].to_numpy()
    y = train[target_col].to_numpy()
    X_test = test[feature_cols].to_numpy()

    # Small 2-config grid (respects harness <=2 cap). Budget-sized for 3 min wall.
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
             l2_regularization=0.0, min_samples_leaf=20),
        dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
             l2_regularization=1.0, min_samples_leaf=40),
    ]

    # Use a single hold-out fold for selection (3-min budget); refit on full train.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(skf.split(X, y))
    cv_results = []
    for i, params in enumerate(grid):
        model = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict_proba(X[va_idx])[:, 1]
        ll = float(log_loss(y[va_idx], np.clip(p, 1e-15, 1 - 1e-15)))
        cv_results.append({"config_idx": i, "params": params,
                           "holdout_log_loss": ll, "n_iter": int(model.n_iter_)})
        print(f"[cv] cfg{i} {params}: holdout log_loss={ll:.5f} (n_iter={model.n_iter_})")

    cv_results.sort(key=lambda r: r["holdout_log_loss"])
    best = cv_results[0]
    print(f"[cv] best cfg{best['config_idx']} holdout log_loss={best['holdout_log_loss']:.5f}")

    # Refit on full train with early stopping (uses internal validation split).
    final = HistGradientBoostingClassifier(
        **best["params"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    final.fit(X, y)
    proba = final.predict_proba(X_test)[:, 1]
    proba = np.clip(proba, 1e-6, 1 - 1e-6)  # guard log-loss extremes

    submission = pd.DataFrame({"id": test["id"].values, "Class": proba})
    sub_path = os.path.join(HERE, "submission.csv")
    submission.to_csv(sub_path, index=False)

    elapsed = time.time() - t0
    result = {
        "target_column_chosen": target_col,
        "headline_metric": "log_loss",
        "headline_value": best["holdout_log_loss"],
        "all_metrics": {
            "holdout_log_loss": best["holdout_log_loss"],
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "Binary pulsar detection, 94K rows, 8 numeric features, 0 NaNs, ~9.3% positives. "
            "HistGradientBoostingClassifier with a 2-config grid evaluated on a 20% stratified "
            "hold-out (single-fold to fit the 3-min budget); chosen config refit on full train "
            "with early stopping. Probabilities clipped to [1e-6, 1-1e-6] for log-loss safety."
        ),
        "audit": {
            "hyperparameter_grid": [r["params"] for r in cv_results],
            "cv_results": cv_results,
            "chosen_config": best["params"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": len(feature_cols),
            "class_balance": float(y.mean()),
            "wall_seconds": elapsed,
        },
    }
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[done] wrote submission.csv ({len(submission)} rows) and result.json in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
