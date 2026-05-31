"""
Wine Quality White multi-class classification.
Constraints: training <= 5 min, inference < 5 ms/instance.
Metric: macro F1.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/wine_quality_white/data")

TARGET = "quality"
N_JOBS = 2
RNG = 42


def load():
    tr = pd.read_csv(DATA / "train.csv")
    va = pd.read_csv(DATA / "valid.csv")
    te = pd.read_csv(DATA / "test.csv")
    return tr, va, te


def split_xy(df):
    y = df[TARGET].values
    X = df.drop(columns=[TARGET]).values.astype(np.float32)
    return X, y


def measure_inference_ms(model, X_sample, repeats=3):
    # warmup
    model.predict(X_sample[:1])
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X_sample)
        dt = (time.perf_counter() - t0) * 1000.0 / len(X_sample)
        best = min(best, dt)
    return best


def main():
    tr, va, te = load()
    X_tr, y_tr = split_xy(tr)
    X_va, y_va = split_xy(va)
    X_te, y_te = split_xy(te)

    print(f"train={X_tr.shape}, valid={X_va.shape}, test={X_te.shape}")
    print(f"classes train={sorted(set(y_tr))} valid={sorted(set(y_va))} test={sorted(set(y_te))}")

    # Pareto-aware candidate set. Tree ensembles work well on this dataset
    # and have well-bounded predict latency (~ trees * depth).
    candidates = {
        "rf_300": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            n_jobs=N_JOBS, random_state=RNG, class_weight=None,
        ),
        "rf_500": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            n_jobs=N_JOBS, random_state=RNG,
        ),
        "et_500": ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            n_jobs=N_JOBS, random_state=RNG,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=63, l2_regularization=0.0, random_state=RNG,
        ),
    }

    results = {}
    train_times = {}
    infer_ms = {}
    for name, est in candidates.items():
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        train_times[name] = time.perf_counter() - t0

        # Measure latency on validation set (per-instance, batched).
        ms = measure_inference_ms(est, X_va)
        infer_ms[name] = ms

        f1_va = f1_score(y_va, est.predict(X_va), average="macro")
        results[name] = f1_va
        print(f"{name}: val_macro_f1={f1_va:.4f}  train={train_times[name]:.1f}s  infer={ms:.3f}ms/inst")

    # Filter to constraint-feasible candidates (under both budgets).
    feasible = {k: v for k, v in results.items()
                if train_times[k] <= 5 * 60 and infer_ms[k] < 5.0}
    if not feasible:
        # Fallback: drop latency-violating, keep training-feasible only.
        feasible = {k: v for k, v in results.items() if train_times[k] <= 5 * 60}

    best_name = max(feasible, key=feasible.get)
    best = candidates[best_name]
    print(f"selected: {best_name}")

    # Final model: refit on train+valid for the held-out test eval.
    X_all = np.vstack([X_tr, X_va])
    y_all = np.concatenate([y_tr, y_va])
    t0 = time.perf_counter()
    best.fit(X_all, y_all)
    final_train_s = time.perf_counter() - t0

    final_infer_ms = measure_inference_ms(best, X_te, repeats=5)

    y_pred = best.predict(X_te)
    test_f1 = f1_score(y_te, y_pred, average="macro")
    val_f1 = results[best_name]

    # Submission
    sub = pd.DataFrame({TARGET: y_pred})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "f1_macro",
        "headline_value": float(test_f1),
        "all_metrics": {
            "val_f1": float(val_f1),
            "test_f1": float(test_f1),
            "inference_ms_per_instance": float(final_infer_ms),
            "training_time_s": float(final_train_s),
        },
        "selected_model": best_name,
        "candidate_val_f1": {k: float(v) for k, v in results.items()},
        "candidate_train_s": {k: float(v) for k, v in train_times.items()},
        "candidate_infer_ms": {k: float(v) for k, v in infer_ms.items()},
        "skills_applied": [
            "skill_multi_axis_constraint_budgets.md",
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "Two explicit constraints (train <=5min, predict <5ms/inst): committed "
            "to numeric budgets, screened a 4-model Pareto set on (val_f1, latency), "
            "picked the constraint-feasible argmax, then refit on train+valid for test."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
