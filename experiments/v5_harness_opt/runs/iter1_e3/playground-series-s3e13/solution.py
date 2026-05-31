"""
Playground Series S3E13 — predict prognosis (11-class) from 64 binary symptom
features. Metric: MAP@3 (space-separated top-3).

Skills applied (from harness skill library):
  * skill_tune_when_n_train_supports_it: n_train=565 >= 500, no
    size/latency/production tokens -> small explicit HGB grid (2 configs).
  * Family-comparison rule (n_train >= 500): run LR and HGB under the SAME
    StratifiedKFold-5; if relative gap < 0.5% average their test predict_proba,
    else pick winner by mean CV.

Not applied:
  * skill_stringify_low_cardinality_int_codes: features are already 0/1 floats.
  * skill_missing_indicator_*: no NaNs in either split.
  * skill_probe_group_structure_in_ids: id is just a sequential integer,
    train/test do not share ids.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e13/data/"
SEED = 42
K = 3
N_SPLITS = 5


def map_at_k(y_true_idx: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(-proba, axis=1)[:, :k]
    score = 0.0
    for i, yt in enumerate(y_true_idx):
        hits = np.where(topk[i] == yt)[0]
        if hits.size:
            score += 1.0 / (hits[0] + 1)
    return score / len(y_true_idx)


def align_proba(model_proba, model_classes, n_classes, idx):
    """Place model_proba (n, len(model_classes)) into full (n, n_classes) at rows idx."""
    full = np.zeros((model_proba.shape[0], n_classes))
    for ci, cls in enumerate(model_classes):
        full[:, cls] = model_proba[:, ci]
    return full


def cv_score(model_fn, X, y, n_classes, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        full = align_proba(p, m.classes_, n_classes, va)
        fold_scores.append(map_at_k(y[va], full, k=K))
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    feat_cols = [c for c in train.columns if c not in ("id", "prognosis")]
    le = LabelEncoder()
    y = le.fit_transform(train["prognosis"].values)
    X = train[feat_cols].values.astype(np.float32)
    Xte = test[feat_cols].values.astype(np.float32)
    classes = le.classes_
    n_classes = len(classes)

    # ----- LR family -----
    def make_lr():
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, n_jobs=1, random_state=SEED)

    lr_mean, lr_std = cv_score(make_lr, X, y, n_classes)

    # ----- HGB family (small grid) -----
    hgb_grid = [
        dict(max_iter=100, learning_rate=0.1,  max_leaf_nodes=31, l2_regularization=0.0),
        dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0),
    ]
    hgb_results = []
    for cfg in hgb_grid:
        def make_hgb(cfg=cfg):
            return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **cfg)
        m, s = cv_score(make_hgb, X, y, n_classes)
        hgb_results.append((m, s, cfg))
    hgb_results.sort(key=lambda r: -r[0])
    hgb_mean, hgb_std, hgb_best_cfg = hgb_results[0]

    def make_hgb_best():
        return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **hgb_best_cfg)

    # ----- Family comparison -----
    rel_gap = abs(lr_mean - hgb_mean) / max(abs(lr_mean), abs(hgb_mean), 1e-9)
    average_families = rel_gap < 0.005

    # ----- Fit on full train, predict test -----
    lr_full = make_lr().fit(X, y)
    hgb_full = make_hgb_best().fit(X, y)
    p_lr = align_proba(lr_full.predict_proba(Xte), lr_full.classes_, n_classes, np.arange(len(Xte)))
    p_hgb = align_proba(hgb_full.predict_proba(Xte), hgb_full.classes_, n_classes, np.arange(len(Xte)))

    if average_families:
        proba = 0.5 * p_lr + 0.5 * p_hgb
        chosen = "average(LR,HGB)"
        headline = (lr_mean + hgb_mean) / 2.0
    elif lr_mean >= hgb_mean:
        proba = p_lr
        chosen = "LR"
        headline = lr_mean
    else:
        proba = p_hgb
        chosen = "HGB"
        headline = hgb_mean

    topk_idx = np.argsort(-proba, axis=1)[:, :K]
    preds = [" ".join(classes[ix] for ix in row) for row in topk_idx]
    sub = pd.DataFrame({"id": test["id"].values, "prognosis": preds})
    sub.to_csv("submission.csv", index=False)

    result = {
        "target_column_chosen": "prognosis",
        "headline_metric": "MAP@3",
        "headline_value": float(headline),
        "all_metrics": {
            "lr_cv_mean": lr_mean,
            "lr_cv_std": lr_std,
            "hgb_cv_mean": hgb_mean,
            "hgb_cv_std": hgb_std,
            "hgb_grid_results": [
                {"cfg": cfg, "mean": m, "std": s} for (m, s, cfg) in hgb_results
            ],
            "family_relative_gap": rel_gap,
            "averaged": average_families,
            "chosen_family": chosen,
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "audit": {
            "family_comparison": {
                "LR": {"mean": lr_mean, "std": lr_std},
                "HGB": {"mean": hgb_mean, "std": hgb_std, "cfg": hgb_best_cfg},
                "rel_gap": rel_gap,
                "averaged": average_families,
            }
        },
        "interpretation_notes": (
            f"11-class prognosis from 64 binary symptoms (n_train=565, n_test=142, no NaNs). "
            f"LR(multinomial) and HGB(2-config tune) under shared StratifiedKFold-5; "
            f"chose '{chosen}' (rel_gap={rel_gap:.4f}, LR={lr_mean:.4f}, HGB={hgb_mean:.4f}). "
            f"MAP@3 with top-3 space-separated labels."
        ),
        "elapsed_sec": time.time() - t0,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
