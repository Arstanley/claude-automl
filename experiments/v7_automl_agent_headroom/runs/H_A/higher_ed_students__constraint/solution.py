"""
Higher Education Students Performance — unsupervised clustering into 8 groups.

Skills applied:
- skill_stringify_low_cardinality_int_codes: all features are integer-coded
  Likert/categorical codes — one-hot encode before clustering.
- skill_multi_axis_constraint_budgets: Rand >= 0.8 AND <= 10 min — try a small
  Pareto-aware set of clusterers + feature views, pick best by Rand on train.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = Path("/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/higher_ed_students/data")
TARGET = "OUTPUT Grade"
N_CLUSTERS = 8
SEED = 0

t0 = time.time()

train = pd.read_csv(DATA / "train.csv")
valid = pd.read_csv(DATA / "valid.csv")
test = pd.read_csv(DATA / "test.csv")

y_train = train[TARGET].to_numpy()
y_valid = valid[TARGET].to_numpy() if TARGET in valid.columns else None

feat_cols = [c for c in train.columns if c != TARGET]
X_train_df = train[feat_cols]
X_valid_df = valid[feat_cols]
X_test_df = test[feat_cols]

X_all_df = pd.concat([X_train_df, X_valid_df, X_test_df], axis=0, ignore_index=True)
n_train = len(X_train_df)
n_valid = len(X_valid_df)


def featurize(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "onehot_all":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return enc.fit_transform(df.astype(str))
    if mode == "onehot_no_course":
        cols = [c for c in df.columns if c != "Course ID"]
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return enc.fit_transform(df[cols].astype(str))
    if mode == "course_id_onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return enc.fit_transform(df[["Course ID"]].astype(str))
    if mode == "course_id_plus_onehot":
        # Course ID heavily weighted: replicate it many times in one-hot space
        enc_all = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_all = enc_all.fit_transform(df.astype(str))
        enc_c = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_c = enc_c.fit_transform(df[["Course ID"]].astype(str))
        # weight Course-ID block by 5x
        return np.hstack([X_all, np.tile(X_c, 5)])
    if mode == "standardized":
        return StandardScaler().fit_transform(df.to_numpy(dtype=float))
    raise ValueError(mode)


candidates = []


def add(name: str, X: np.ndarray, model_fn, k: int = N_CLUSTERS):
    t = time.time()
    try:
        labels_all = model_fn(X)
    except Exception as e:  # noqa: BLE001
        candidates.append((name, -1.0, -1.0, None, str(e)))
        return
    lt = labels_all[:n_train]
    rs = rand_score(y_train, lt)
    ars = adjusted_rand_score(y_train, lt)
    candidates.append((name, rs, ars, labels_all, f"{time.time()-t:.2f}s k={k}"))


feature_views = {
    "onehot_all": featurize(X_all_df, "onehot_all"),
    "onehot_no_course": featurize(X_all_df, "onehot_no_course"),
    "course_only": featurize(X_all_df, "course_id_onehot"),
    "course_weighted": featurize(X_all_df, "course_id_plus_onehot"),
}

# Sweep k around 8 (target is 8 clusters; with k=9 we can still score Rand)
for view_name, X_view in feature_views.items():
    for k in (8,):
        add(
            f"kmeans/{view_name}/k{k}",
            X_view,
            lambda X, k=k: KMeans(n_clusters=k, n_init=30, random_state=SEED).fit_predict(X),
            k=k,
        )
        for lk in ("ward", "average", "complete"):
            add(
                f"agg_{lk}/{view_name}/k{k}",
                X_view,
                lambda X, k=k, lk=lk: AgglomerativeClustering(n_clusters=k, linkage=lk).fit_predict(X),
                k=k,
            )
        add(
            f"gmm/{view_name}/k{k}",
            X_view,
            lambda X, k=k: GaussianMixture(
                n_components=k, covariance_type="diag", random_state=SEED, n_init=3, reg_covar=1e-3
            ).fit_predict(X),
            k=k,
        )
        try:
            add(
                f"spectral/{view_name}/k{k}",
                X_view,
                lambda X, k=k: SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    n_neighbors=10,
                    assign_labels="kmeans",
                    random_state=SEED,
                ).fit_predict(X),
                k=k,
            )
        except Exception:
            pass

# Baseline: use Course ID directly as the cluster label (it has 9 unique values,
# remap to 8 by KMeans-style merging is overkill — just use the codes).
course_all = X_all_df["Course ID"].astype("category").cat.codes.to_numpy()
candidates.append(
    (
        "course_id_direct",
        rand_score(y_train, course_all[:n_train]),
        adjusted_rand_score(y_train, course_all[:n_train]),
        course_all,
        "0.00s k=9",
    )
)

# Sort by Rand on train
candidates.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)
print(f"\n{'Candidate':50s} {'rand':>7s} {'adj_rand':>9s}  info")
for name, rs, ars, _, info in candidates[:15]:
    print(f"  {name:50s} {rs:7.4f} {ars:9.4f}  {info}")

best_name, best_rand, best_ars, best_labels, best_info = candidates[0]

val_rand = None
if y_valid is not None and best_labels is not None:
    val_rand = float(rand_score(y_valid, best_labels[n_train : n_train + n_valid]))

# Submission for test rows
test_labels = best_labels[n_train + n_valid :]
submission = pd.DataFrame({"OUTPUT Grade": test_labels.astype(int)})
submission.to_csv(HERE / "submission.csv", index=False)

elapsed = time.time() - t0
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "rand_score_train",
    "headline_value": float(best_rand),
    "all_metrics": {
        "rand_score_train": float(best_rand),
        "adjusted_rand_score_train": float(best_ars),
        "rand_score_valid": val_rand,
        "best_model": best_name,
        "elapsed_seconds": elapsed,
        "top_candidates": [
            {"name": n, "rand": float(r), "adj_rand": float(a)}
            for (n, r, a, _, _) in candidates[:10]
        ],
    },
    "skills_applied": [
        "skill_stringify_low_cardinality_int_codes.md",
        "skill_multi_axis_constraint_budgets.md",
    ],
    "interpretation_notes": (
        "All 31 features are low-cardinality integer codes; one-hot encoded the joint"
        " (train+valid+test) feature matrix. Sweeped KMeans / Agglomerative (ward/avg/complete)"
        " / GMM / Spectral over four feature views (all, no-course, course-only, course-weighted)"
        f" and a Course-ID-direct baseline. Best: {best_name} (train Rand={best_rand:.4f},"
        f" adj-Rand={best_ars:.4f})."
    ),
}

with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(
    f"\nBest: {best_name}  rand={best_rand:.4f}  adj_rand={best_ars:.4f}"
    f"  valid_rand={val_rand}  elapsed={elapsed:.1f}s"
)
print(f"Wrote submission.csv ({len(submission)} rows) and result.json")
