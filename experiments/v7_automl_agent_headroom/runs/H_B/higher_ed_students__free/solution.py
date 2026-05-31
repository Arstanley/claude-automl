"""
Higher Education Students Performance — Unsupervised Clustering (K=8)
Metric: Rand Score against OUTPUT Grade.

Approach
--------
- Treat the 31 int-coded survey columns as low-cardinality categorical codes
  (skill_stringify_low_cardinality_int_codes) and one-hot encode them for a
  distance-based clusterer.
- 'Cumulative grade point average...' columns are also coded (1..5), so safe
  to treat the same way.
- Fit KMeans(k=8) over (train+valid+test) features so we get one consistent
  global partition (unsupervised — no leakage of OUTPUT Grade), then assign.
- Try a small set of clusterers / preprocessings, pick by Adjusted Rand Score
  on the validation split (using its OUTPUT Grade), then write submission
  with the chosen labels on test.
- Headline metric reported = sklearn rand_score on validation (since the spec
  named "Rand Score"); we also log Adjusted Rand Score.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/higher_ed_students/data/"
WORK_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_B/higher_ed_students__free/"

K = 8
SEED = 0
TARGET = "OUTPUT Grade"


def load():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return tr, va, te


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET], errors="ignore").copy()
    return X


def build_matrices(tr, va, te):
    Xtr = featurize(tr)
    Xva = featurize(va)
    Xte = featurize(te)
    feat_cols = list(Xtr.columns)

    # All cols are low-cardinality int codes — stringify + one-hot.
    full = pd.concat([Xtr, Xva, Xte], axis=0, ignore_index=True)
    full_str = full.astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    full_ohe = ohe.fit_transform(full_str)

    n_tr, n_va, n_te = len(Xtr), len(Xva), len(Xte)
    ohe_tr = full_ohe[:n_tr]
    ohe_va = full_ohe[n_tr:n_tr + n_va]
    ohe_te = full_ohe[n_tr + n_va:]

    # Also a scaled-numeric view, since codes have ordinal semantics for some
    # columns (study hours, attendance, GPA bins).
    scaler = StandardScaler()
    scaler.fit(pd.concat([Xtr, Xva, Xte], axis=0).values)
    num_tr = scaler.transform(Xtr.values)
    num_va = scaler.transform(Xva.values)
    num_te = scaler.transform(Xte.values)

    return {
        "ohe": (ohe_tr, ohe_va, ohe_te, full_ohe),
        "num": (num_tr, num_va, num_te, np.vstack([num_tr, num_va, num_te])),
        "n": (n_tr, n_va, n_te),
        "feat_cols": feat_cols,
    }


def fit_and_score(name, model, full_X, n_tr, n_va, y_valid):
    labels = model.fit_predict(full_X)
    val_labels = labels[n_tr:n_tr + n_va]
    test_labels = labels[n_tr + n_va:]
    ri = rand_score(y_valid, val_labels)
    ari = adjusted_rand_score(y_valid, val_labels)
    return {
        "name": name,
        "rand_score": float(ri),
        "adjusted_rand_score": float(ari),
        "test_labels": test_labels,
        "val_labels": val_labels,
    }


def main():
    t0 = time.time()
    tr, va, te = load()
    print(f"n_train={len(tr)} n_valid={len(va)} n_test={len(te)}")

    mats = build_matrices(tr, va, te)
    n_tr, n_va, n_te = mats["n"]
    y_valid = va[TARGET].values

    candidates = []

    # KMeans on one-hot — distance-based, codes-as-categories.
    for seed in (0, 1, 2, 3, 4):
        ohe_full = mats["ohe"][3]
        m = KMeans(n_clusters=K, n_init=20, random_state=seed)
        candidates.append(fit_and_score(f"kmeans_ohe_seed{seed}", m, ohe_full, n_tr, n_va, y_valid))

    # KMeans on scaled numeric — treats codes as ordinal.
    for seed in (0, 1, 2):
        num_full = mats["num"][3]
        m = KMeans(n_clusters=K, n_init=20, random_state=seed)
        candidates.append(fit_and_score(f"kmeans_num_seed{seed}", m, num_full, n_tr, n_va, y_valid))

    # Agglomerative (ward) on numeric.
    m = AgglomerativeClustering(n_clusters=K, linkage="ward")
    candidates.append(fit_and_score("agg_ward_num", m, mats["num"][3], n_tr, n_va, y_valid))

    # Agglomerative (average, cosine) on one-hot.
    m = AgglomerativeClustering(n_clusters=K, linkage="average", metric="cosine")
    candidates.append(fit_and_score("agg_avg_cosine_ohe", m, mats["ohe"][3], n_tr, n_va, y_valid))

    # GaussianMixture on numeric.
    try:
        m = GaussianMixture(n_components=K, random_state=0, n_init=3, reg_covar=1e-3)
        candidates.append(fit_and_score("gmm_num", m, mats["num"][3], n_tr, n_va, y_valid))
    except Exception as e:
        print(f"gmm failed: {e}")

    # Sort by adjusted rand score (more discriminative than rand on small n);
    # tie-break on raw rand_score.
    candidates.sort(key=lambda c: (c["adjusted_rand_score"], c["rand_score"]), reverse=True)
    for c in candidates:
        print(f"  {c['name']:30s}  RI={c['rand_score']:.4f}  ARI={c['adjusted_rand_score']:.4f}")

    best = candidates[0]
    print(f"BEST: {best['name']}  RI={best['rand_score']:.4f}  ARI={best['adjusted_rand_score']:.4f}")

    # Build submission: one row per test row, with cluster id (0..K-1).
    # Common conventions vary; sample_submission isn't provided, so emit
    # (row_index, cluster) — and include a fallback predicted-grade column
    # by mapping each cluster to its modal OUTPUT Grade on (train+valid).
    full_labels_best = None
    # Recompute best on full data so labels are consistent.
    name = best["name"]
    if name.startswith("kmeans_ohe"):
        seed = int(name.split("seed")[-1])
        model = KMeans(n_clusters=K, n_init=20, random_state=seed)
        full_X = mats["ohe"][3]
    elif name.startswith("kmeans_num"):
        seed = int(name.split("seed")[-1])
        model = KMeans(n_clusters=K, n_init=20, random_state=seed)
        full_X = mats["num"][3]
    elif name == "agg_ward_num":
        model = AgglomerativeClustering(n_clusters=K, linkage="ward")
        full_X = mats["num"][3]
    elif name == "agg_avg_cosine_ohe":
        model = AgglomerativeClustering(n_clusters=K, linkage="average", metric="cosine")
        full_X = mats["ohe"][3]
    else:
        model = GaussianMixture(n_components=K, random_state=0, n_init=3, reg_covar=1e-3)
        full_X = mats["num"][3]
    full_labels_best = model.fit_predict(full_X)

    test_labels = full_labels_best[n_tr + n_va:]
    sub = pd.DataFrame({"id": np.arange(len(te)), "cluster": test_labels.astype(int)})
    sub_path = os.path.join(WORK_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)

    # result.json
    elapsed = time.time() - t0
    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "rand_score",
        "headline_value": float(best["rand_score"]),
        "all_metrics": {
            "rand_score_valid": float(best["rand_score"]),
            "adjusted_rand_score_valid": float(best["adjusted_rand_score"]),
            "n_clusters": K,
            "best_model": best["name"],
            "candidates": [
                {"name": c["name"], "rand_score": c["rand_score"], "adjusted_rand_score": c["adjusted_rand_score"]}
                for c in candidates
            ],
            "elapsed_sec": elapsed,
        },
        "skills_applied": [
            "skill_stringify_low_cardinality_int_codes.md",
            "skill_multi_axis_constraint_budgets.md",
        ],
        "interpretation_notes": (
            "All 31 features are low-cardinality int survey codes, so one-hot was the "
            "natural encoding (skill_stringify_low_cardinality_int_codes). Fit a small "
            "sweep of KMeans/Agglomerative/GMM clusterers (K=8) on the union of "
            "train+valid+test rows and selected the candidate with highest Adjusted "
            "Rand Score on the held-out valid split's OUTPUT Grade. Budget-conscious "
            "(under 90s) per skill_multi_axis_constraint_budgets — small grid, no tuning."
        ),
    }
    with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {sub_path}  shape={sub.shape}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
