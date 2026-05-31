"""
Playground Series S3E13 — Vector Borne Disease Prediction (MPA@3).

Multiclass classification (11 prognoses) on 64 binary symptom features.
Metric: Mean Precision @ 3. Submission: top-3 labels per row, space-separated.

Skills applied:
- skill_tune_when_n_train_supports_it: n_train=565>=500, GBM family, no size/latency
  constraint in prompt -> small explicit grid with stratified CV.

Strategy:
- Use scikit-learn's LogisticRegression (good baseline for binary features +
  multiclass) and an ensemble with GradientBoosting-style models is overkill on
  64 binary features with only 565 rows. We tune a small grid on
  LogisticRegression (C) and RandomForest (n_estimators / max_depth) with
  stratified 5-fold CV scoring MPA@3 directly. Pick the best, refit on full
  train, predict probabilities on test, take top-3 by probability.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e13/data/")
WORK_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/runs/H_0/playground-series-s3e13/")
WORK_DIR.mkdir(parents=True, exist_ok=True)

RNG = 42
K_TOP = 3


def mpa_at_k(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int = 3) -> float:
    """Mean Average Precision @ k for single-label multiclass.

    Score = sum_{i=1..k} (1/i) * [pred_i == true] but with the convention used
    by Kaggle MPA@K: AP@K = sum_{i=1..k} P(i)*rel(i) / min(K, n_relevant) where
    n_relevant=1 for single-label, so AP@K = (1/rank) if true is in top-k else 0.
    """
    # rank of true class in descending proba
    # proba shape: (n, n_classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    n = len(y_true)
    score = 0.0
    # argsort proba descending
    order = np.argsort(-proba, axis=1)[:, :k]
    for i in range(n):
        true_col = cls_idx.get(y_true[i], -1)
        if true_col < 0:
            continue
        topk = order[i]
        for rank, col in enumerate(topk, start=1):
            if col == true_col:
                score += 1.0 / rank
                break
    return score / n


def cv_score(model_factory, X, y, classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        # ensure class ordering matches `classes`
        proba = m.predict_proba(X[va_idx])
        # align if needed
        if not np.array_equal(m.classes_, classes):
            # remap columns
            col_map = [list(m.classes_).index(c) if c in m.classes_ else -1 for c in classes]
            aligned = np.zeros((proba.shape[0], len(classes)))
            for j, k in enumerate(col_map):
                if k >= 0:
                    aligned[:, j] = proba[:, k]
            proba = aligned
        scores.append(mpa_at_k(y[va_idx], proba, classes, k=K_TOP))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = [c for c in train.columns if c not in ("id", "prognosis")]
    X = train[feature_cols].values.astype(np.float32)
    y_raw = train["prognosis"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = np.arange(len(le.classes_))  # integer class labels
    class_names = le.classes_

    Xt = test[feature_cols].values.astype(np.float32)

    # ---- Small explicit grid ----
    candidates = []

    # Logistic Regression: C grid
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        candidates.append((
            f"logreg_C{C}",
            lambda C=C: LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs",
                multi_class="multinomial", random_state=RNG,
            ),
        ))

    # Random Forest: small grid
    for n_est, max_d in [(300, None), (500, None), (500, 10)]:
        candidates.append((
            f"rf_n{n_est}_d{max_d}",
            lambda n_est=n_est, max_d=max_d: RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d,
                n_jobs=-1, random_state=RNG, class_weight="balanced_subsample",
            ),
        ))

    results = {}
    best_name = None
    best_score = -np.inf
    best_factory = None
    for name, factory in candidates:
        mean, std = cv_score(factory, X, y, classes, n_splits=5)
        results[name] = {"cv_mpa3_mean": mean, "cv_mpa3_std": std}
        print(f"  {name}: MPA@3 = {mean:.4f} +/- {std:.4f}")
        if mean > best_score:
            best_score = mean
            best_name = name
            best_factory = factory

    print(f"Best: {best_name} -> {best_score:.4f}")

    # ---- Also try an average ensemble of best logreg + best rf ----
    # Identify best of each family
    best_lr_name = max(
        (n for n in results if n.startswith("logreg")),
        key=lambda n: results[n]["cv_mpa3_mean"],
    )
    best_rf_name = max(
        (n for n in results if n.startswith("rf_")),
        key=lambda n: results[n]["cv_mpa3_mean"],
    )

    # CV the average ensemble
    def ensemble_cv():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
        scores = []
        lr_factory = next(f for n, f in candidates if n == best_lr_name)
        rf_factory = next(f for n, f in candidates if n == best_rf_name)
        for tr_idx, va_idx in skf.split(X, y):
            m1 = lr_factory(); m1.fit(X[tr_idx], y[tr_idx])
            m2 = rf_factory(); m2.fit(X[tr_idx], y[tr_idx])
            p1 = m1.predict_proba(X[va_idx])
            p2 = m2.predict_proba(X[va_idx])
            # align
            def align(p, m):
                if np.array_equal(m.classes_, classes):
                    return p
                col_map = [list(m.classes_).index(c) for c in classes]
                return p[:, col_map]
            p1 = align(p1, m1); p2 = align(p2, m2)
            p = 0.5 * p1 + 0.5 * p2
            scores.append(mpa_at_k(y[va_idx], p, classes, k=K_TOP))
        return float(np.mean(scores)), float(np.std(scores))

    ens_mean, ens_std = ensemble_cv()
    results["ensemble_lr_rf"] = {"cv_mpa3_mean": ens_mean, "cv_mpa3_std": ens_std}
    print(f"  ensemble_lr_rf: MPA@3 = {ens_mean:.4f} +/- {ens_std:.4f}")

    # Pick final
    use_ensemble = ens_mean >= best_score
    final_choice = "ensemble_lr_rf" if use_ensemble else best_name
    final_score = ens_mean if use_ensemble else best_score
    print(f"Final choice: {final_choice} (CV MPA@3 = {final_score:.4f})")

    # ---- Refit on full train and predict on test ----
    if use_ensemble:
        lr_factory = next(f for n, f in candidates if n == best_lr_name)
        rf_factory = next(f for n, f in candidates if n == best_rf_name)
        m1 = lr_factory(); m1.fit(X, y)
        m2 = rf_factory(); m2.fit(X, y)
        def align(p, m):
            if np.array_equal(m.classes_, classes):
                return p
            col_map = [list(m.classes_).index(c) for c in classes]
            return p[:, col_map]
        p1 = align(m1.predict_proba(Xt), m1)
        p2 = align(m2.predict_proba(Xt), m2)
        proba = 0.5 * p1 + 0.5 * p2
    else:
        m = best_factory(); m.fit(X, y)
        proba = m.predict_proba(Xt)
        if not np.array_equal(m.classes_, classes):
            col_map = [list(m.classes_).index(c) for c in classes]
            proba = proba[:, col_map]

    # top-3 labels per row, space-separated
    top3_idx = np.argsort(-proba, axis=1)[:, :K_TOP]
    top3_labels = np.array([
        " ".join(class_names[idx] for idx in row) for row in top3_idx
    ])

    sub = pd.DataFrame({"id": test["id"].values, "prognosis": top3_labels})
    sub.to_csv(WORK_DIR / "submission.csv", index=False)
    print(f"Wrote submission.csv: {sub.shape}")

    result = {
        "target_column_chosen": "prognosis",
        "headline_metric": "MPA@3 (CV)",
        "headline_value": final_score,
        "all_metrics": {
            "cv_mpa3_by_model": {n: r["cv_mpa3_mean"] for n, r in results.items()},
            "cv_mpa3_std_by_model": {n: r["cv_mpa3_std"] for n, r in results.items()},
            "final_choice": final_choice,
            "best_logreg": best_lr_name,
            "best_rf": best_rf_name,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_classes": int(len(class_names)),
            "elapsed_sec": time.time() - t0,
        },
        "skills_applied": ["skill_tune_when_n_train_supports_it.md"],
        "interpretation_notes": (
            "Multiclass (11 classes) on 565 rows with 64 binary symptom features and "
            "MPA@3 as the metric. Ran a small explicit grid over LogisticRegression "
            "(multinomial, C in {0.1,0.5,1,2,5}) and RandomForest (n_estimators/max_depth) "
            "with 5-fold stratified CV scoring MPA@3 directly, then optionally averaged "
            "the best of each family. Predictions are top-3 class labels per row."
        ),
    }
    (WORK_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote result.json. Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
