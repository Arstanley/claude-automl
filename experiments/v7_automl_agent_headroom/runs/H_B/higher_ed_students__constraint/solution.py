"""
Higher Ed Students Performance - unsupervised clustering (k=8) vs OUTPUT Grade.
Target Rand Index >= 0.8 on test set.

Approach:
- Features are mostly low-cardinality ordinal/nominal int codes; two are 4-pt GPAs.
- Stringify low-cardinality int codes (skill_stringify_low_cardinality_int_codes) + one-hot
  so distance/centroid-based clustering doesn't treat code "5" as 5x "1".
- Scale the resulting features and try several clusterers (KMeans, Agglomerative variants,
  Spectral, GaussianMixture). Pick by train Rand Index against OUTPUT Grade.
- Cluster the union train+valid+test (transductive) so test predictions come from the same
  cluster assignment. This is fine here because the task is purely unsupervised and the
  Rand Index is invariant to label permutation.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score

DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/higher_ed_students/data"
OUT_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/runs/H_B/higher_ed_students__constraint"
TARGET = "OUTPUT Grade"
K = 8
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
valid = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"train={train.shape} valid={valid.shape} test={test.shape}")

# --- skill_stringify_low_cardinality_int_codes ---
# Almost every feature is int code with <= ~10 unique. Two GPA cols are 4-point ordinal
# (values 1..5) but they encode rank; we treat them as numeric to keep order.
GPA_COLS = [
    "Cumulative grade point average in the last semester (/4.00)",
    "Expected Cumulative grade point average in the graduation (/4.00)",
]
feat_cols = [c for c in train.columns if c != TARGET]
cat_cols = [c for c in feat_cols if c not in GPA_COLS]
num_cols = [c for c in feat_cols if c in GPA_COLS]
stringified = list(cat_cols)
print(f"cat={len(cat_cols)} num={len(num_cols)}")

all_df = pd.concat([train[feat_cols], valid[feat_cols], test[feat_cols]], axis=0,
                   ignore_index=True)
for c in cat_cols:
    all_df[c] = all_df[c].astype(str)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols),
])
X_all = pre.fit_transform(all_df)
print(f"feature matrix: {X_all.shape}")

n_train = len(train)
n_valid = len(valid)
n_test = len(test)
X_train = X_all[:n_train]
X_valid = X_all[n_train:n_train + n_valid]
X_test = X_all[n_train + n_valid:]
y_train = train[TARGET].values
y_valid = valid[TARGET].values

# Dimensionality reduction helps small-n high-dim clustering.
candidates = []

def try_clusterer(name, labels_all):
    lt = labels_all[:n_train]
    lv = labels_all[n_train:n_train + n_valid]
    ri_t = rand_score(y_train, lt)
    ri_v = rand_score(y_valid, lv) if len(y_valid) else float("nan")
    ari_t = adjusted_rand_score(y_train, lt)
    nmi_t = normalized_mutual_info_score(y_train, lt)
    candidates.append({
        "name": name,
        "ri_train": float(ri_t),
        "ri_valid": float(ri_v),
        "ari_train": float(ari_t),
        "nmi_train": float(nmi_t),
        "labels_all": labels_all,
    })
    print(f"{name:40s}  RI_train={ri_t:.4f}  RI_valid={ri_v:.4f}  ARI={ari_t:.4f}  NMI={nmi_t:.4f}")


# Try clustering on raw features and on a few PCA projections.
for pca_dim in [None, 2, 5, 10, 20, 30]:
    if pca_dim is None:
        Xc = X_all
        tag = "raw"
    else:
        if pca_dim >= X_all.shape[1]:
            continue
        Xc = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X_all)
        tag = f"pca{pca_dim}"

    # KMeans, several seeds
    for seed in [0, 1, 2, 7, 13]:
        km = KMeans(n_clusters=K, n_init=20, random_state=seed)
        labs = km.fit_predict(Xc)
        try_clusterer(f"kmeans_{tag}_s{seed}", labs)

    # Agglomerative variants
    for linkage in ["ward", "average", "complete"]:
        try:
            ac = AgglomerativeClustering(n_clusters=K, linkage=linkage)
            labs = ac.fit_predict(Xc)
            try_clusterer(f"agglo_{tag}_{linkage}", labs)
        except Exception as e:
            print(f"agglo_{tag}_{linkage} failed: {e}")

    # GaussianMixture
    for cov in ["full", "tied", "diag"]:
        try:
            gm = GaussianMixture(
                n_components=K, covariance_type=cov,
                random_state=SEED, n_init=5, reg_covar=1e-3,
            )
            labs = gm.fit_predict(Xc)
            try_clusterer(f"gmm_{tag}_{cov}", labs)
        except Exception as e:
            print(f"gmm_{tag}_{cov} failed: {e}")

# Spectral (only on raw; expensive but small n)
for affinity in ["rbf", "nearest_neighbors"]:
    try:
        sc = SpectralClustering(
            n_clusters=K, affinity=affinity, random_state=SEED,
            assign_labels="kmeans", n_neighbors=10,
        )
        labs = sc.fit_predict(X_all)
        try_clusterer(f"spectral_{affinity}", labs)
    except Exception as e:
        print(f"spectral_{affinity} failed: {e}")

# Pick best by train RI (unsupervised task: we have access to OUTPUT Grade on train
# for model selection per task spec "checked against target variable OUTPUT Grade").
best = max(candidates, key=lambda d: d["ri_train"])
print(f"\nBEST: {best['name']}  RI_train={best['ri_train']:.4f}  RI_valid={best['ri_valid']:.4f}")

best_labels_all = best["labels_all"]
test_labels = best_labels_all[n_train + n_valid:]

# Submission
test_with_id = test.reset_index().rename(columns={"index": "id"})
sub = pd.DataFrame({"id": test_with_id["id"], "cluster": test_labels})
sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(f"wrote submission.csv rows={len(sub)}")

# If we have test labels available (they live in the test.csv as OUTPUT Grade),
# also compute a real test RI for the result.json.
y_test = test[TARGET].values if TARGET in test.columns else None
ri_test = float(rand_score(y_test, test_labels)) if y_test is not None else None
ari_test = float(adjusted_rand_score(y_test, test_labels)) if y_test is not None else None

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "rand_score",
    "headline_value": ri_test if ri_test is not None else best["ri_train"],
    "all_metrics": {
        "ri_train": best["ri_train"],
        "ri_valid": best["ri_valid"],
        "ari_train": best["ari_train"],
        "nmi_train": best["nmi_train"],
        "ri_test": ri_test,
        "ari_test": ari_test,
    },
    "skills_applied": [
        "skill_stringify_low_cardinality_int_codes.md",
        "skill_multi_axis_constraint_budgets.md",
    ],
    "interpretation_notes": (
        f"Unsupervised clustering with k={K}. Stringified all 29 low-cardinality int-coded "
        f"feature columns and one-hot encoded them, leaving the two 4-point GPA columns as "
        f"scaled numeric. Compared KMeans / Agglomerative / GaussianMixture / Spectral over "
        f"raw and PCA projections, selected by Rand Index against OUTPUT Grade on train. "
        f"Best: {best['name']}."
    ),
    "audit": {
        "stringified_columns": stringified,
        "best_method": best["name"],
        "n_train": int(n_train), "n_valid": int(n_valid), "n_test": int(n_test),
        "feature_dim": int(X_all.shape[1]),
        "n_candidates_tried": len(candidates),
    },
}

with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2, default=lambda o: None)
print(f"wrote result.json — headline={result['headline_value']}")
