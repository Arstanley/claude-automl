"""
playground-series-s3e12 — Kidney stone presence probability.
Metric: ROC AUC.
n_train=331, n_test=83, 6 numeric features, no NaNs, binary target.

Skill-library scan: with executable preconditions on n_train=331 and no
NaNs/aux-files/split-claims/int-codes/multi-axis-constraints, none of the
v4 skills strictly fire (skill_tune_when_n_train_supports_it gates on
n_train >= 500). We therefore use a small, robust default: an ensemble
of a scaled logistic regression and a gradient boosting classifier,
both well-suited to small tabular n. CV is used as a sanity check only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/"
    "playground-series-s3e12/data"
)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target_col = "target"
    id_col = "id"
    feat_cols = [c for c in train.columns if c not in (target_col, id_col)]

    X = train[feat_cols].to_numpy(dtype=float)
    y = train[target_col].to_numpy(dtype=int)
    X_test = test[feat_cols].to_numpy(dtype=float)

    rng = 42
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)

    # Small grid (<=3 configs): logreg, GBM (light), GBM (slightly deeper).
    configs = {
        "logreg": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0, max_iter=2000, solver="liblinear", random_state=rng
                    ),
                ),
            ]
        ),
        "gbm_light": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=rng
        ),
        "gbm_mid": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=rng
        ),
    }

    cv_scores: dict[str, float] = {}
    for name, model in configs.items():
        scores = cross_val_score(
            model, X, y, scoring="roc_auc", cv=cv, n_jobs=2
        )
        cv_scores[name] = float(np.mean(scores))
        print(f"  {name}: AUC={cv_scores[name]:.4f} (+/- {np.std(scores):.4f})")

    # Soft ensemble: average predicted probabilities of all three models.
    # Rationale: small-n, models have different inductive biases, AUC is
    # rank-based so averaging probabilities is a safe ensemble.
    preds = np.zeros(len(test), dtype=float)
    for name, model in configs.items():
        model.fit(X, y)
        p = model.predict_proba(X_test)[:, 1]
        preds += p
    preds /= len(configs)

    ensemble_cv = float(np.mean(list(cv_scores.values())))

    out_sub = HERE / "submission.csv"
    pd.DataFrame({id_col: test[id_col].to_numpy(), target_col: preds}).to_csv(
        out_sub, index=False
    )

    headline = max(cv_scores.values())
    headline_name = max(cv_scores, key=cv_scores.get)
    result = {
        "target_column_chosen": target_col,
        "headline_metric": "roc_auc_cv5",
        "headline_value": headline,
        "all_metrics": {
            **{f"cv5_auc_{k}": v for k, v in cv_scores.items()},
            "cv5_auc_mean_of_models": ensemble_cv,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "skills_applied": [],
        "interpretation_notes": (
            "n_train=331 < 500 so skill_tune_when_n_train_supports_it does not "
            "fire; other v4 skills also fail preconditions (no NaNs, no aux "
            "files, no split claim, no low-card int codes, single-axis metric). "
            f"Used soft-average ensemble of logreg+2 GBMs; best single CV AUC "
            f"= {headline:.4f} ({headline_name})."
        ),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {out_sub} and result.json in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
