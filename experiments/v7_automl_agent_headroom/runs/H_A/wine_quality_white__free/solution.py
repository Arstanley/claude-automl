"""
Wine Quality White - multi-class (ordinal-coded) macro-F1 classification.

Skills applied:
- skill_tune_when_n_train_supports_it (n_train=2938, no size/latency constraints, GBM family):
    small explicit grid over HistGradientBoostingClassifier + RandomForest, picked
    by macro-F1 on the provided valid.csv split.
- skill_train_on_listed_auxiliary_files (prompt lists train AND valid as available):
    after picking the best config on train -> valid, refit on train+valid before
    predicting test.

Outputs: submission.csv (id,quality), result.json.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/wine_quality_white/data")
TARGET = "quality"
N_JOBS = 2
SEED = 42

t0 = time.time()


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    valid = pd.read_csv(DATA / "valid.csv")
    test = pd.read_csv(DATA / "test.csv")

    feat_cols = [c for c in train.columns if c != TARGET]
    X_tr, y_tr = train[feat_cols].values, train[TARGET].values
    X_va, y_va = valid[feat_cols].values, valid[TARGET].values
    X_te = test[feat_cols].values
    has_test_label = TARGET in test.columns
    y_te = test[TARGET].values if has_test_label else None

    print(f"shapes: train={X_tr.shape}  valid={X_va.shape}  test={X_te.shape}")
    print(f"train classes: {sorted(np.unique(y_tr).tolist())}")

    # ---- small explicit grid ---------------------------------------------------
    # Mix of HGB (well-suited for small tabular) + RF (variance reduction).
    # NB: HGB max_depth must be capped — unlimited depth on 7 classes blows up
    # (per-class tree per iter; ~7 * max_iter trees, each grows fully on 2.9k rows).
    configs: list[tuple[str, object]] = [
        (
            "hgb_lr05_d6",
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_depth=6,
                min_samples_leaf=20, l2_regularization=0.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=SEED,
            ),
        ),
        (
            "hgb_lr05_d8_l2_1",
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_depth=8,
                min_samples_leaf=10, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=SEED,
            ),
        ),
        (
            "hgb_lr1_d6",
            HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=300, max_depth=6,
                min_samples_leaf=20, l2_regularization=0.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=SEED,
            ),
        ),
        (
            "rf_400",
            RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=1,
                max_features="sqrt", random_state=SEED, n_jobs=N_JOBS,
            ),
        ),
        (
            "rf_600_balanced",
            RandomForestClassifier(
                n_estimators=600, max_depth=None, min_samples_leaf=1,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=SEED, n_jobs=N_JOBS,
            ),
        ),
    ]

    results: list[dict] = []
    for name, est in configs:
        t = time.time()
        est.fit(X_tr, y_tr)
        pred_va = est.predict(X_va)
        score = macro_f1(y_va, pred_va)
        results.append({"name": name, "valid_macro_f1": float(score), "fit_s": time.time() - t})
        print(f"  {name:24s}  valid macro-F1 = {score:.4f}  ({time.time()-t:.1f}s)")

    results.sort(key=lambda r: r["valid_macro_f1"], reverse=True)
    best_name = results[0]["name"]
    best_valid = results[0]["valid_macro_f1"]
    print(f"best on valid: {best_name} = {best_valid:.4f}")

    # ---- refit best on train + valid (auxiliary file skill) -------------------
    X_full = np.vstack([X_tr, X_va])
    y_full = np.concatenate([y_tr, y_va])

    best_cfg = dict(configs)[best_name]
    # re-instantiate (sklearn refit is fine, but be explicit)
    best_cfg.fit(X_full, y_full)
    pred_te = best_cfg.predict(X_te)

    # ---- write submission ------------------------------------------------------
    if "id" in test.columns:
        sub = pd.DataFrame({"id": test["id"].values, TARGET: pred_te})
    else:
        sub = pd.DataFrame({"id": np.arange(len(pred_te)), TARGET: pred_te})
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  rows={len(sub)}")

    # ---- score on test if labels present (sanity only) -----------------------
    test_macro_f1 = None
    test_acc = None
    if has_test_label:
        test_macro_f1 = float(macro_f1(y_te, pred_te))
        test_acc = float((pred_te == y_te).mean())
        print(f"(test labels present)  macro-F1={test_macro_f1:.4f}  acc={test_acc:.4f}")

    # ---- result.json -----------------------------------------------------------
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "macro_f1",
        "headline_value": test_macro_f1 if test_macro_f1 is not None else best_valid,
        "all_metrics": {
            "valid_macro_f1_best": best_valid,
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_acc,
            "grid_results": results,
            "best_config": best_name,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
            "skill_train_on_listed_auxiliary_files.md",
        ],
        "interpretation_notes": (
            "Tabular 11-feature ordinal-coded multi-class. Ran a 5-config GBM/RF "
            "grid scored on the held-out valid.csv, then refit the winner on "
            "train+valid before predicting test. No missing values, no scaling "
            "needed (tree models). Class 3/9 are tiny so macro-F1 is dominated "
            "by minority-class recall."
        ),
        "elapsed_s": time.time() - t0,
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {HERE / 'result.json'}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
