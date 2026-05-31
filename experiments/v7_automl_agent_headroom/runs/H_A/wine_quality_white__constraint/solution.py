"""
Wine Quality White - multi-class ordinal classification.
Metric: macro F1. Constraints: train <=5min, inference <=5ms/instance.

Skills applied:
  - skill_multi_axis_constraint_budgets.md
    Multiple axes (train-time, per-instance latency, accuracy). We commit
    to concrete numeric budgets (300s train, 5ms/inst inference) and do a
    Pareto-aware small model search (HGBM vs RF), picking the highest
    valid macro-F1 that clears the latency budget.

Skipped:
  - skill_train_on_listed_auxiliary_files: winequality-white.csv is the
    union of train+valid+test (4898 = 2938+735+1225), so concatenating
    would leak test labels. Also, the prompt does not name it.
  - skill_tune_when_n_train_supports_it: precondition fails -- the prompt
    explicitly contains latency/size tokens ("minutes", "milliseconds"),
    so the v2-misfire suppressor activates. Stick with a small explicit
    candidate set, no big grid search.
  - skill_missing_indicator_*: zero NaN.
  - skill_stringify_low_cardinality_int_codes: no int feature columns.
"""
from __future__ import annotations

import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/wine_quality_white/data"
TARGET = "quality"
N_JOBS_FIT = 2
LATENCY_BUDGET_MS = 5.0
TRAIN_BUDGET_S = 300
RNG = 0


def per_instance_latency_ms(model, X, n_single=30):
    """Average single-row predict latency. Suppress n_jobs to avoid
    parallelization overhead dominating one-row inference."""
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    # warm up
    model.predict(X.iloc[:1])
    rows = X.iloc[: min(n_single, len(X))]
    t0 = time.perf_counter()
    for i in range(len(rows)):
        model.predict(rows.iloc[i : i + 1])
    return (time.perf_counter() - t0) * 1000.0 / len(rows)


def main():
    t_global = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    valid = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    feat_cols = [c for c in train.columns if c != TARGET]
    X_tr, y_tr = train[feat_cols], train[TARGET].astype(int)
    X_va, y_va = valid[feat_cols], valid[TARGET].astype(int)
    X_te = test[feat_cols]
    y_te = test[TARGET].astype(int) if TARGET in test.columns else None

    # Small candidate set sized to the latency budget.
    # Pilot measurements (n_jobs=1 inference) on this dataset:
    #   RF n_estimators=100 -> ~4ms/inst, F1 ~0.37
    #   RF n_estimators=150 -> ~5.5ms/inst, F1 ~0.38 (over budget)
    #   RF n_estimators=200 -> ~7ms/inst   (over budget)
    #   HGBM -> 30-65ms/inst (single-row predict overhead dominates) -> always over.
    # We therefore explore inside the budget band only.
    candidates = {
        "rf_100": RandomForestClassifier(
            n_estimators=100,
            n_jobs=N_JOBS_FIT,
            random_state=RNG,
            class_weight="balanced_subsample",
        ),
        "rf_120": RandomForestClassifier(
            n_estimators=120,
            n_jobs=N_JOBS_FIT,
            random_state=RNG,
            class_weight="balanced_subsample",
        ),
        # Include HGBM as a reference point; we expect it to be over-budget on
        # single-row latency but it documents the Pareto frontier.
        "hgbm_ref": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31, random_state=RNG
        ),
    }

    leaderboard = []
    fitted = {}
    for name, mdl in candidates.items():
        if time.time() - t_global > TRAIN_BUDGET_S - 30:
            print(f"[skip {name}] train budget nearly exhausted")
            break
        t0 = time.time()
        mdl.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        pred_va = mdl.predict(X_va)
        f1 = f1_score(y_va, pred_va, average="macro")
        lat_ms = per_instance_latency_ms(mdl, X_va)
        leaderboard.append(
            {
                "name": name,
                "valid_macro_f1": float(f1),
                "fit_s": float(fit_s),
                "latency_ms_per_instance": float(lat_ms),
            }
        )
        fitted[name] = mdl
        print(
            f"[{name}] fit={fit_s:.1f}s valid_macroF1={f1:.4f} lat={lat_ms:.2f}ms/inst"
        )

    # Pareto-aware selection: highest valid macro-F1 among models that meet
    # the latency budget. If none meet it, prefer the fastest (constraint-
    # respecting) model -- HGBM essentially always wins here.
    eligible = [r for r in leaderboard if r["latency_ms_per_instance"] <= LATENCY_BUDGET_MS]
    if eligible:
        best_rec = max(eligible, key=lambda r: r["valid_macro_f1"])
        constraint_met = True
    else:
        best_rec = min(leaderboard, key=lambda r: r["latency_ms_per_instance"])
        constraint_met = False
    best_name = best_rec["name"]
    print(
        f"\nSelected {best_name} (latency-constraint met: {constraint_met})"
    )

    # Refit on train+valid with same hyperparams for final test predictions.
    X_full = pd.concat([X_tr, X_va], ignore_index=True)
    y_full = pd.concat([y_tr, y_va], ignore_index=True)
    final_cls = candidates[best_name].__class__
    final_params = candidates[best_name].get_params()
    final_model = final_cls(**final_params)
    final_model.fit(X_full, y_full)

    # Bulk test prediction + per-instance latency on the test set itself.
    if hasattr(final_model, "n_jobs"):
        final_model.n_jobs = 1
    t0 = time.perf_counter()
    pred_te = final_model.predict(X_te)
    bulk_ms = (time.perf_counter() - t0) * 1000.0
    bulk_per_instance_ms = bulk_ms / len(X_te)
    single_lat_ms = per_instance_latency_ms(final_model, X_te)

    submission = pd.DataFrame({TARGET: pred_te.astype(int)})
    submission.to_csv(os.path.join(HERE, "submission.csv"), index=False)

    metrics = {"valid_macro_f1": float(best_rec["valid_macro_f1"])}
    if y_te is not None:
        metrics["test_macro_f1"] = float(f1_score(y_te, pred_te, average="macro"))
        metrics["test_accuracy"] = float((pred_te == y_te.values).mean())
        print("\n--- test classification report ---")
        print(classification_report(y_te, pred_te, zero_division=0))

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "macro_f1",
        "headline_value": metrics.get("test_macro_f1", metrics["valid_macro_f1"]),
        "all_metrics": metrics,
        "leaderboard": leaderboard,
        "selected_model": best_name,
        "selected_model_params": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in final_params.items()
        },
        "final_single_row_latency_ms": float(single_lat_ms),
        "final_bulk_latency_ms_per_instance": float(bulk_per_instance_ms),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "latency_budget_met": bool(single_lat_ms <= LATENCY_BUDGET_MS),
        "train_budget_s": TRAIN_BUDGET_S,
        "wall_clock_s": float(time.time() - t_global),
        "n_train": int(len(X_tr)),
        "n_valid": int(len(X_va)),
        "n_test": int(len(X_te)),
        "skills_applied": ["skill_multi_axis_constraint_budgets.md"],
        "skills_considered_and_skipped": {
            "skill_train_on_listed_auxiliary_files.md": "winequality-white.csv is the union of train+valid+test (would leak); prompt does not name it",
            "skill_tune_when_n_train_supports_it.md": "prompt contains latency/time tokens (ms, minutes) -> v2 misfire suppressor active",
        },
        "interpretation_notes": (
            "Pareto-aware model selection across HGBM and RF under explicit "
            "train-time (300s) and per-instance latency (5ms) budgets; "
            "selected the model with highest valid macro-F1 that clears "
            "the latency budget, then refit on train+valid for test."
        ),
    }
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"\nHeadline macro F1: {result['headline_value']:.4f} | "
        f"single-row lat: {single_lat_ms:.2f}ms | wall: {result['wall_clock_s']:.1f}s"
    )


if __name__ == "__main__":
    main()
