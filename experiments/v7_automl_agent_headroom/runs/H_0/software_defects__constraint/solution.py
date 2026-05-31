"""Software Defects binary classification (F1) under constraint:
- train <= 30 min, inference <= 5 ms/instance
- HistGradientBoostingClassifier + F1-tuned threshold on holdout.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent.resolve()
DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/software_defects/data")
TARGET = "defects"
ID = "id"


def main() -> None:
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")

    y = train_df[TARGET].astype(int).values
    feat_cols = [c for c in train_df.columns if c not in (TARGET, ID)]
    X = train_df[feat_cols].values
    X_test = test_df[feat_cols].values

    # Holdout split for threshold tuning + reporting
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    t0 = time.time()
    # Light grid (constraint skill: stay well under budget)
    candidates = [
        dict(max_depth=None, max_iter=400, learning_rate=0.06, l2_regularization=0.0),
        dict(max_depth=8, max_iter=500, learning_rate=0.05, l2_regularization=0.1),
        dict(max_depth=6, max_iter=600, learning_rate=0.05, l2_regularization=0.0),
    ]
    best = None
    for cfg in candidates:
        clf = HistGradientBoostingClassifier(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            **cfg,
        )
        clf.fit(X_tr, y_tr)
        prob_va = clf.predict_proba(X_va)[:, 1]
        # tune threshold on validation
        ths = np.linspace(0.1, 0.9, 81)
        f1s = [f1_score(y_va, (prob_va >= t).astype(int)) for t in ths]
        j = int(np.argmax(f1s))
        f1_best = f1s[j]
        th_best = float(ths[j])
        if best is None or f1_best > best["f1"]:
            best = {"clf": clf, "f1": f1_best, "th": th_best, "cfg": cfg}
    train_time = time.time() - t0

    # Refit best config on full train for final predictions
    final = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        **best["cfg"],
    )
    t1 = time.time()
    final.fit(X, y)
    refit_time = time.time() - t1

    # Inference timing
    t2 = time.time()
    prob_test = final.predict_proba(X_test)[:, 1]
    inf_total = time.time() - t2
    inf_ms_per = 1000.0 * inf_total / max(1, len(X_test))

    # Re-tune threshold on holdout from the held-out fit; reuse best threshold
    th = best["th"]
    pred = (prob_test >= th).astype(int)

    sub = pd.DataFrame({ID: test_df[ID].values, TARGET: pred})
    sub.to_csv(HERE / "submission.csv", index=False)

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "f1",
        "headline_value": float(best["f1"]),
        "all_metrics": {
            "f1_holdout": float(best["f1"]),
            "threshold": float(th),
            "training_time_s": float(train_time + refit_time),
            "inference_ms_per_instance": float(inf_ms_per),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "pos_rate_train": float(y.mean()),
            "pred_pos_rate_test": float(pred.mean()),
            "best_config": best["cfg"],
        },
        "skills_applied": [
            "skill_multi_axis_constraint_budgets.md",
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingClassifier (handles NaN natively, vectorized inference well "
            "under 5ms/instance). Tuned 3-config grid with early stopping; F1 threshold "
            "selected on stratified holdout to address ~23% positive imbalance."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("done:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
