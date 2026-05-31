"""
playground-series-s3e2 — Stroke probability prediction.

Task: binary classification, metric = ROC AUC, submission = (id, P(stroke=1)).
Data: ~12k train rows, ~3k test rows, 10 features (mix categorical / numeric),
heavily imbalanced (~4% positive). No missing values. No id overlap train/test.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=12243 >= 500, prompt has no
  size/latency/MB/ms/production tokens -> tune a small explicit GBM grid.

Skills evaluated and skipped:
- skill_missing_indicator_when_missingness_signals_class: per-column NaN rate
  is 0% on every column -> N/A.
- skill_probe_group_structure_in_ids: id overlap between train & test = 0;
  features are demographic/clinical, no compound IDs -> N/A; use StratifiedKFold.
- skill_stringify_low_cardinality_int_codes: model is tree-based (HGB), so
  binary int columns (hypertension, heart_disease) need no special casting.
- skill_verify_split_claim_before_holdout: prompt makes no structural split claim.
- skill_train_on_listed_auxiliary_files: original Stroke Prediction Dataset is
  *mentioned* as available but not present in DATA_DIR -> can't apply.
- skill_multi_axis_constraint_budgets: single-axis (AUC) -> N/A.

Pipeline: HistGradientBoostingClassifier with native categorical support, small
3-config grid tuned with 5-fold stratified CV on ROC AUC, refit on full train.
class_weight='balanced' to handle imbalance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e2/data")

TARGET = "stroke"
ID = "id"
CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status",
            "hypertension", "heart_disease"]
NUM_COLS = ["age", "avg_glucose_level", "bmi"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    return tr, te


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # cast categoricals (object + binary ints) to pandas 'category' so HGB
    # picks them up natively
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df[CAT_COLS + NUM_COLS]


def main() -> None:
    t0 = time.time()
    train_df, test_df = load()
    print(f"train: {train_df.shape}  test: {test_df.shape}")
    print(f"target dist: {train_df[TARGET].value_counts().to_dict()}")

    y = train_df[TARGET].to_numpy()
    X = prep(train_df)
    X_test = prep(test_df)
    assert list(X.columns) == list(X_test.columns)

    # small explicit grid (skill_tune_when_n_train_supports_it)
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
             l2_regularization=0.0, min_samples_leaf=20),
        dict(max_iter=500, learning_rate=0.03, max_depth=None, max_leaf_nodes=31,
             l2_regularization=1.0, min_samples_leaf=20),
        dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=63,
             l2_regularization=1.0, min_samples_leaf=40),
        # smaller / regularized variant, useful when target is rare
        dict(max_iter=600, learning_rate=0.02, max_depth=None, max_leaf_nodes=15,
             l2_regularization=1.0, min_samples_leaf=50),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    results = []
    for params in grid:
        model = HistGradientBoostingClassifier(
            **params,
            categorical_features=[X.columns.get_loc(c) for c in CAT_COLS],
            class_weight="balanced",
            early_stopping=False,
            random_state=0,
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=5)
        results.append((scores.mean(), scores.std(), params, scores.tolist()))
        print(f"  {params}: AUC = {scores.mean():.5f} ± {scores.std():.5f}")

    results.sort(key=lambda r: r[0], reverse=True)
    best_mean, best_std, best_params, best_scores = results[0]
    print(f"\nbest: {best_params}  AUC={best_mean:.5f} ± {best_std:.5f}")

    # Also evaluate a simpler default-ish baseline for sanity comparison
    baseline = HistGradientBoostingClassifier(
        categorical_features=[X.columns.get_loc(c) for c in CAT_COLS],
        class_weight="balanced",
        random_state=0,
    )
    baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring="roc_auc", n_jobs=5)
    print(f"baseline (HGB defaults): AUC = {baseline_scores.mean():.5f} ± {baseline_scores.std():.5f}")

    # Refit best config on full train
    final = HistGradientBoostingClassifier(
        **best_params,
        categorical_features=[X.columns.get_loc(c) for c in CAT_COLS],
        class_weight="balanced",
        early_stopping=False,
        random_state=0,
    )
    final.fit(X, y)

    # Predict test probabilities
    test_proba = final.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({ID: test_df[ID].to_numpy(), TARGET: test_proba})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  ({len(sub)} rows)")

    # Sanity: in-sample AUC (not a true estimate, just to confirm pipeline runs)
    train_proba = final.predict_proba(X)[:, 1]
    in_sample_auc = roc_auc_score(y, train_proba)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "roc_auc",
        "headline_value": float(best_mean),
        "all_metrics": {
            "cv_auc_mean": float(best_mean),
            "cv_auc_std": float(best_std),
            "cv_auc_fold_scores": [float(s) for s in best_scores],
            "baseline_cv_auc_mean": float(baseline_scores.mean()),
            "baseline_cv_auc_std": float(baseline_scores.std()),
            "in_sample_auc_after_refit": float(in_sample_auc),
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Binary stroke prediction, AUC metric, ~4% positive class. Used "
            "HistGradientBoosting with native categorical handling (object + "
            "binary-int columns cast to category), class_weight='balanced' for "
            "imbalance, and a 4-config grid tuned with 5-fold StratifiedKFold "
            "on AUC. Best config chosen by CV mean; refit on full train."
        ),
        "audit": {
            "n_train": int(len(X)),
            "n_test": int(len(X_test)),
            "cat_cols": CAT_COLS,
            "num_cols": NUM_COLS,
            "hyperparameter_grid": [
                {"params": p, "cv_auc_mean": float(m), "cv_auc_std": float(s),
                 "fold_scores": [float(x) for x in fs]}
                for m, s, p, fs in results
            ],
            "chosen_config": best_params,
            "wallclock_s": round(time.time() - t0, 1),
        },
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {HERE / 'result.json'}")
    print(f"headline AUC = {best_mean:.5f}  (in {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
