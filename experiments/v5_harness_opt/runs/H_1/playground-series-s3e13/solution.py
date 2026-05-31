"""Solution for Kaggle Playground S3E13: vector-borne disease prognosis prediction.

Task: multi-class (11) prediction with MPA@3 (Mean Precision at 3) metric.
Output format: space-separated top-3 class labels per row.

Approach:
- Tabular: 64 binary symptom features (already 0/1 float), no NaN, n_train=565.
- Skill applied: skill_tune_when_n_train_supports_it (n_train >= 500, no size constraints)
  -> small explicit grid of GBM configurations chosen by CV MPA@3.
- Also fit a Logistic Regression as a complement (cheap second family) and blend
  by OOF; this is informational since blend skill precondition (n>=1000) not met
  -> we still try the LR blend if CV improves, otherwise ship single model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e13/data")

K = 3  # MPA@K
N_JOBS = 2
SEED = 42


def mpa_at_k(y_true, top_k_pred):
    """Mean Precision at K: average precision over rows.

    For each row, score = (1/rank) if true label appears in top-k predictions
    (rank = 1..K), else 0. Then mean across rows. This matches Kaggle's MAP@K
    definition for the case of 1 ground-truth label per row, which Kaggle
    refers to as MPA@K here.
    """
    y_true = np.asarray(y_true)
    score = 0.0
    n = len(y_true)
    for i in range(n):
        preds = top_k_pred[i]
        for rank, p in enumerate(preds[:K], start=1):
            if p == y_true[i]:
                score += 1.0 / rank
                break
    return score / n


def top_k_from_proba(proba, classes, k=K):
    """Return array of shape (n, k) of top-k class labels per row."""
    idx = np.argsort(-proba, axis=1)[:, :k]
    return classes[idx]


def cv_score(model_factory, X, y, classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(X), len(classes)))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        # Align proba columns to global `classes` order
        local_classes = m.classes_
        p = m.predict_proba(X[va_idx])
        for j, c in enumerate(local_classes):
            oof_proba[va_idx, np.where(classes == c)[0][0]] = p[:, j]
    top = top_k_from_proba(oof_proba, classes, K)
    return mpa_at_k(y, top), oof_proba


def main():
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    target = "prognosis"
    feats = [c for c in train.columns if c not in {"id", target}]
    X = train[feats].to_numpy(dtype=np.float32)
    y = train[target].to_numpy()
    X_test = test[feats].to_numpy(dtype=np.float32)
    classes = np.array(sorted(train[target].unique()))

    print(f"n_train={len(X)}, n_test={len(X_test)}, n_features={len(feats)}, "
          f"n_classes={len(classes)}")

    # ------------------------------------------------------------------
    # Skill: tune_when_n_train_supports_it -> small grid over RF + GBM
    # We use RandomForest (handles small-n binary features well, fast) and
    # HistGradientBoostingClassifier as the GBM. Tiny explicit grid.
    # ------------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingClassifier

    cv_results = []

    # Random Forest grid
    rf_grid = [
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 800, "max_depth": 12,   "min_samples_leaf": 1},
    ]
    for cfg in rf_grid:
        def mk(cfg=cfg):
            return RandomForestClassifier(
                random_state=SEED, n_jobs=N_JOBS, **cfg
            )
        s, _ = cv_score(mk, X, y, classes)
        cv_results.append(("rf", cfg, s))
        print(f"RF {cfg} -> CV MPA@3 = {s:.4f}")

    # HistGBM grid
    hgb_grid = [
        {"max_iter": 300, "learning_rate": 0.05, "max_depth": None, "max_leaf_nodes": 31},
        {"max_iter": 500, "learning_rate": 0.03, "max_depth": None, "max_leaf_nodes": 15},
    ]
    for cfg in hgb_grid:
        def mk(cfg=cfg):
            return HistGradientBoostingClassifier(random_state=SEED, **cfg)
        s, _ = cv_score(mk, X, y, classes)
        cv_results.append(("hgb", cfg, s))
        print(f"HGB {cfg} -> CV MPA@3 = {s:.4f}")

    # Logistic regression baseline (also acts as second family for blend trial)
    def mk_lr():
        return LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", n_jobs=N_JOBS, random_state=SEED
        )
    s_lr, oof_lr = cv_score(mk_lr, X, y, classes)
    cv_results.append(("lr", {"C": 1.0}, s_lr))
    print(f"LR  C=1.0 -> CV MPA@3 = {s_lr:.4f}")

    # Pick best single model
    cv_results.sort(key=lambda r: r[2], reverse=True)
    best_family, best_cfg, best_score = cv_results[0]
    print(f"\nBest single: {best_family} {best_cfg} -> {best_score:.4f}")

    # Refit best with OOF for potential blending
    if best_family == "rf":
        best_factory = lambda: RandomForestClassifier(
            random_state=SEED, n_jobs=N_JOBS, **best_cfg
        )
    elif best_family == "hgb":
        best_factory = lambda: HistGradientBoostingClassifier(
            random_state=SEED, **best_cfg
        )
    else:
        best_factory = mk_lr

    best_cv_score, oof_best = cv_score(best_factory, X, y, classes)

    # Attempt simple LR blend (informational, since n<1000 -> blend skill not gated on).
    blend_used = False
    final_oof = oof_best
    best_weight = 1.0
    if best_family != "lr":
        # search blend weight
        weights = [0.0, 0.1, 0.2, 0.3, 0.5]
        for w in weights:
            blend = (1 - w) * oof_best + w * oof_lr
            top = top_k_from_proba(blend, classes, K)
            s = mpa_at_k(y, top)
            if s > best_cv_score + 1e-6:
                best_cv_score = s
                best_weight = 1 - w
                final_oof = blend
                blend_used = (w > 0)
        print(f"Blend weight on best={best_weight:.2f}, blended? {blend_used}, "
              f"CV={best_cv_score:.4f}")

    # ------------------------------------------------------------------
    # Final fit on full train + predict on test
    # ------------------------------------------------------------------
    final_model = best_factory()
    final_model.fit(X, y)
    proba_best = np.zeros((len(X_test), len(classes)))
    for j, c in enumerate(final_model.classes_):
        proba_best[:, np.where(classes == c)[0][0]] = final_model.predict_proba(X_test)[:, j]

    if blend_used:
        lr_final = mk_lr()
        lr_final.fit(X, y)
        proba_lr = np.zeros((len(X_test), len(classes)))
        for j, c in enumerate(lr_final.classes_):
            proba_lr[:, np.where(classes == c)[0][0]] = lr_final.predict_proba(X_test)[:, j]
        proba_test = best_weight * proba_best + (1 - best_weight) * proba_lr
    else:
        proba_test = proba_best

    top_test = top_k_from_proba(proba_test, classes, K)

    submission = pd.DataFrame({
        "id": test["id"].values,
        "prognosis": [" ".join(row) for row in top_test],
    })
    sub_path = HERE / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nWrote {sub_path}, head:")
    print(submission.head().to_string(index=False))

    # ------------------------------------------------------------------
    # result.json
    # ------------------------------------------------------------------
    skills_applied = [
        "skill_tune_when_n_train_supports_it.md",
    ]
    skills_evaluated_not_applied = [
        "skill_probe_group_structure_in_ids.md: 0 ID overlap and no compound IDs -> precondition fails",
        "skill_train_on_listed_auxiliary_files.md: prompt mentions an external original dataset but no auxiliary file is shipped in DATA_DIR -> precondition fails",
        "skill_verify_split_claim_before_holdout.md: no structural split claim made by prompt -> precondition fails",
        "skill_stringify_low_cardinality_int_codes.md: features are 0/1 binary floats already; chosen models are tree-based (no scaler/linear path needs encoding) -> precondition fails",
        "skill_multi_axis_constraint_budgets.md: only the MPA@3 metric is constrained, no latency/size axis -> precondition fails",
        "skill_missing_indicator_when_missingness_signals_class.md: no NaN in any column -> precondition fails",
        "skill_tune_decision_threshold_for_f1.md: multi-class (11), not binary; metric is MPA@3 not F1 -> precondition fails",
        "skill_log_target_for_skewed_nonnegative_regression.md: classification task, not regression -> precondition fails",
        "skill_blend_two_model_families.md: n_train=565 < 1000 -> precondition fails (we did try a cheap LR blend opportunistically; only kept if CV improved)",
    ]

    result = {
        "target_column_chosen": target,
        "headline_metric": "MPA@3",
        "headline_value": float(best_cv_score),
        "all_metrics": {
            "cv_mpa_at_3_best_single": float(cv_results[0][2]),
            "cv_mpa_at_3_final": float(best_cv_score),
            "blend_used": bool(blend_used),
            "blend_weight_on_best": float(best_weight),
            "best_family": best_family,
            "best_cfg": best_cfg,
            "cv_grid": [
                {"family": f, "cfg": c, "cv_mpa_at_3": float(s)}
                for f, c, s in cv_results
            ],
        },
        "skills_applied": skills_applied,
        "skills_evaluated_not_applied": skills_evaluated_not_applied,
        "interpretation_notes": (
            "Small (n=565) multi-class tabular task with 64 binary symptom features and "
            "11 classes scored by MPA@3. Tuned a small grid of RF + HistGBM + LR and "
            "selected the best by 5-fold stratified CV MPA@3. Did not apply the blend "
            "skill (n<1000) though we tried an opportunistic LR blend kept only if "
            "CV improved."
        ),
        "runtime_seconds": float(time.time() - t0),
    }
    res_path = HERE / "result.json"
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {res_path}")
    print(f"Total runtime: {result['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
