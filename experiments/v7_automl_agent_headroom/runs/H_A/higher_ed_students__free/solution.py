"""
Unsupervised clustering of higher-ed students into 8 groups.
Scored by Adjusted Rand Index (and raw Rand Index) vs OUTPUT Grade.

Approach
--------
- Concatenate train + valid + test (unsupervised, no leakage worries).
- All features are integer category codes (ordinals 1..N) per UCI 856 dataset.
  Apply skill_stringify_low_cardinality_int_codes: one-hot encode them so
  KMeans Euclidean distance is meaningful (not pretending code 5 > code 1).
- Run KMeans n_clusters=8 with many seeds (n_init=50) and pick best inertia.
- Also try Gaussian Mixture and Agglomerative; pick the one with best
  internal silhouette to assign final cluster IDs on the TEST rows.
- Score against OUTPUT Grade column on test rows (the label we cluster
  against, but never use during fitting).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

t0 = time.time()

DATA_DIR = Path(
    "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/higher_ed_students/data"
)
WORK_DIR = Path(
    "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_A/higher_ed_students__free"
)
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "OUTPUT Grade"
N_CLUSTERS = 8
RNG = 0

train = pd.read_csv(DATA_DIR / "train.csv")
valid = pd.read_csv(DATA_DIR / "valid.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"shapes: train={train.shape} valid={valid.shape} test={test.shape}")

# Concatenate everything; clustering is unsupervised.
all_df = pd.concat([train, valid, test], axis=0, ignore_index=True)
feature_cols = [c for c in all_df.columns if c != TARGET]

# Skill: stringify low-cardinality int codes, then one-hot.  Every column here
# is an integer with <=11 unique values across 147 rows -> all qualify.
X_codes = all_df[feature_cols].astype(str)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X = encoder.fit_transform(X_codes)
print(f"one-hot feature matrix: {X.shape}")

# Slice indices for test rows
n_train, n_valid, n_test = len(train), len(valid), len(test)
test_slice = slice(n_train + n_valid, n_train + n_valid + n_test)
y_test_true = test[TARGET].to_numpy()


def score(labels: np.ndarray) -> dict:
    test_labels = labels[test_slice]
    return {
        "rand_score": float(rand_score(y_test_true, test_labels)),
        "adjusted_rand_score": float(adjusted_rand_score(y_test_true, test_labels)),
    }


candidates = {}

# --- KMeans with many seeds ---
km = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=RNG)
km_labels = km.fit_predict(X)
sil_km = silhouette_score(X, km_labels) if len(set(km_labels)) > 1 else -1
candidates["kmeans"] = (km_labels, sil_km)

# --- Agglomerative (Ward) on dense one-hot ---
try:
    agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    agg_labels = agg.fit_predict(X)
    sil_agg = silhouette_score(X, agg_labels) if len(set(agg_labels)) > 1 else -1
    candidates["agglomerative_ward"] = (agg_labels, sil_agg)
except Exception as e:  # noqa: BLE001
    print(f"agglomerative failed: {e}")

# --- Gaussian Mixture ---
try:
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="diag",  # robust on wide one-hot
        n_init=5,
        random_state=RNG,
        max_iter=200,
    )
    gmm_labels = gmm.fit_predict(X)
    sil_gmm = silhouette_score(X, gmm_labels) if len(set(gmm_labels)) > 1 else -1
    candidates["gmm_diag"] = (gmm_labels, sil_gmm)
except Exception as e:  # noqa: BLE001
    print(f"gmm failed: {e}")

# Report all candidate scores for transparency.
all_metrics = {}
for name, (labels, sil) in candidates.items():
    s = score(labels)
    s["silhouette"] = float(sil)
    all_metrics[name] = s
    print(f"{name:>20s}: rand={s['rand_score']:.4f} ari={s['adjusted_rand_score']:.4f} sil={sil:.4f}")

# Pick best by silhouette (unsupervised model selection — no peeking at labels).
best_name = max(candidates, key=lambda k: candidates[k][1])
best_labels = candidates[best_name][0]
print(f"selected (by silhouette): {best_name}")

final_test_labels = best_labels[test_slice]
final = score(best_labels)

# --- Submission ---
sub = pd.DataFrame({"cluster": final_test_labels.astype(int)})
sub.to_csv(WORK_DIR / "submission.csv", index=False)

# --- result.json ---
result = {
    "target_column_chosen": TARGET,
    "headline_metric": "rand_score",
    "headline_value": final["rand_score"],
    "all_metrics": {
        "selected_model": best_name,
        "test_rand_score": final["rand_score"],
        "test_adjusted_rand_score": final["adjusted_rand_score"],
        "per_candidate": all_metrics,
    },
    "skills_applied": [
        "skill_stringify_low_cardinality_int_codes.md",
    ],
    "interpretation_notes": (
        "All 31 features are integer ordinal codes with <=11 unique values; "
        "stringified+one-hot before KMeans/Ward/GMM so Euclidean distance "
        "respects category identity rather than ordinal magnitude. Model "
        "selected unsupervised by silhouette; rand_score reported on test "
        "rows vs OUTPUT Grade."
    ),
    "audit": {
        "stringified_columns": feature_cols,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "n_clusters": N_CLUSTERS,
        "runtime_s": round(time.time() - t0, 2),
    },
}
with open(WORK_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"DONE in {time.time() - t0:.1f}s  rand={final['rand_score']:.4f}  ari={final['adjusted_rand_score']:.4f}")
