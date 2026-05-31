"""Wine Quality White — multi-class (quality 3-9), metric: macro F1.

Strategy
--------
- ~2.9k train rows, 11 numeric features, ordinal int target.
- Skill applied: skill_tune_when_n_train_supports_it (n_train >= 500, GBM family,
  no size/latency tokens in prompt) -> small explicit grid of strong tabular models.
- Use valid.csv strictly for model selection; only touch test.csv once at the end.
- n_jobs=2, 2-minute budget -> short grid.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/wine_quality_white/data"
)
TARGET = "quality"
N_JOBS = 2
RNG = 42

t0 = time.time()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA_DIR / "train.csv")
    va = pd.read_csv(DATA_DIR / "valid.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    return tr, va, te


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET].astype(int)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def main() -> dict:
    train, valid, test = load()
    X_tr, y_tr = split_xy(train)
    X_va, y_va = split_xy(valid)
    X_te, y_te = split_xy(test)

    # Small explicit grid — RF + ExtraTrees + HGBM, varied a bit.
    candidates: list[tuple[str, object]] = [
        (
            "rf_400_depth_None",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=N_JOBS,
                random_state=RNG,
                class_weight=None,
            ),
        ),
        (
            "rf_600_balanced",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=N_JOBS,
                random_state=RNG,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "et_600",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=N_JOBS,
                random_state=RNG,
            ),
        ),
        (
            "et_800_balanced",
            ExtraTreesClassifier(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=N_JOBS,
                random_state=RNG,
                class_weight="balanced",
            ),
        ),
        (
            "hgb_lr0.05_leaves63",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=63,
                max_iter=500,
                l2_regularization=0.0,
                random_state=RNG,
            ),
        ),
        (
            "hgb_lr0.1_leaves31",
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_leaf_nodes=31,
                max_iter=400,
                random_state=RNG,
            ),
        ),
    ]

    val_results: dict[str, float] = {}
    best_name: str | None = None
    best_val_f1 = -1.0
    best_model = None

    for name, model in candidates:
        if time.time() - t0 > 95:  # leave ~25s for refit+predict+write
            print(f"[skip] time budget cutoff before {name}")
            break
        t1 = time.time()
        model.fit(X_tr, y_tr)
        pred_va = model.predict(X_va)
        f1v = macro_f1(y_va, pred_va)
        val_results[name] = f1v
        print(f"[val] {name}: macro_f1={f1v:.4f} (fit+pred {time.time()-t1:.1f}s)")
        if f1v > best_val_f1:
            best_val_f1 = f1v
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    # Refit selected model on train+valid for final test predictions.
    X_full = pd.concat([X_tr, X_va], axis=0, ignore_index=True)
    y_full = pd.concat([y_tr, y_va], axis=0, ignore_index=True)
    # clone-ish: refit the same estimator object (sklearn refits in place)
    best_model.fit(X_full, y_full)

    pred_te = best_model.predict(X_te)
    test_f1 = macro_f1(y_te, pred_te)
    test_acc = float(accuracy_score(y_te, pred_te))

    # Submission file.
    sub = pd.DataFrame({TARGET: pred_te.astype(int)})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "f1_macro",
        "headline_value": test_f1,
        "all_metrics": {
            "val_f1": best_val_f1,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
        },
        "model_selected": best_name,
        "val_grid": val_results,
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Tabular 11-feature numeric, n_train ~2.9k, ordinal int target treated "
            "as multi-class. Ran a short explicit grid over RF / ExtraTrees / "
            "HistGradientBoosting, selected on valid macro-F1, refit on train+valid "
            "for final test prediction."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"total elapsed: {time.time()-t0:.1f}s")
    return result


if __name__ == "__main__":
    main()
