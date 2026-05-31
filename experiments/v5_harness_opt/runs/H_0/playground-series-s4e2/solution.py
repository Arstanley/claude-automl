"""Solution for playground-series-s4e2: multi-class obesity classification.

Task: predict NObeyesdad (7-class categorical) from demographic/lifestyle features.
Metric: accuracy.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=16606 >> 500 and no constraint
  tokens in the prompt, so we run a small explicit grid over HistGradientBoosting
  configs (sklearn-only env, no xgboost/lightgbm available).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).resolve().parent
DATA = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s4e2/data"
)

CAT_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
NUM_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
TARGET = "NObeyesdad"


def build_pipeline(**hgb_kwargs) -> Pipeline:
    # Ordinal-encode categoricals; HGB tree splits handle ordinal codes fine,
    # and we mark them as categorical so HGB uses categorical-aware splits.
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                CAT_COLS,
            ),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    cat_mask = [True] * len(CAT_COLS) + [False] * len(NUM_COLS)
    model = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=0,
        **hgb_kwargs,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    X = train[CAT_COLS + NUM_COLS]
    y = train[TARGET].values
    X_test = test[CAT_COLS + NUM_COLS]

    print(f"train shape: {train.shape}, test shape: {test.shape}")
    print(f"classes: {sorted(set(y))}")

    # Small explicit grid (skill_tune_when_n_train_supports_it).
    grid = [
        dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31),
        dict(max_iter=600, learning_rate=0.05, max_depth=None, max_leaf_nodes=63),
        dict(max_iter=800, learning_rate=0.03, max_depth=None, max_leaf_nodes=31),
        dict(max_iter=400, learning_rate=0.08, max_depth=None, max_leaf_nodes=63),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    results = []
    for params in grid:
        pipe = build_pipeline(**params)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=5)
        mean, std = float(scores.mean()), float(scores.std())
        print(f"  {params}: acc = {mean:.4f} +/- {std:.4f}")
        results.append((mean, std, params))

    results.sort(key=lambda r: r[0], reverse=True)
    best_mean, best_std, best_params = results[0]
    print(f"\nbest: {best_params} -> {best_mean:.4f} +/- {best_std:.4f}")

    # Refit on full train.
    final = build_pipeline(**best_params).fit(X, y)
    preds = final.predict(X_test)

    submission = pd.DataFrame({"id": test["id"].values, TARGET: preds})
    submission.to_csv(HERE / "submission.csv", index=False)
    print(f"wrote submission: {submission.shape}")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "accuracy",
        "headline_value": best_mean,
        "all_metrics": {
            "cv_accuracy_mean": best_mean,
            "cv_accuracy_std": best_std,
            "grid_results": [
                {"params": p, "mean": m, "std": s} for (m, s, p) in results
            ],
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "Multi-class (7-way) tabular classification, n_train=16606, no constraint "
            "tokens in prompt -> applied skill_tune_when_n_train_supports_it with a 4-config "
            "HistGradientBoosting grid (sklearn-only environment; no XGBoost/LightGBM). "
            "Categoricals ordinal-encoded with HGB's native categorical_features handling; "
            "no NaNs so missing-indicator skill did not apply."
        ),
        "chosen_config": best_params,
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("wrote result.json")


if __name__ == "__main__":
    main()
