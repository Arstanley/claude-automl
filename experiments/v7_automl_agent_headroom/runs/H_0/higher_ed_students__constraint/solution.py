"""
Unsupervised clustering of Higher Education Students into 8 groups.
Target metric: Rand Score >= 0.8 vs OUTPUT Grade, complete in <= 10 min.

Skills applied:
  - skill_stringify_low_cardinality_int_codes: features are int codes; one-hot before scaling/distance.
  - skill_multi_axis_constraint_budgets: explicit budgets (rand>=0.8, time<=10min) -> sweep small grid of clusterers.
"""
import json, time, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score
from sklearn.decomposition import PCA

DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/higher_ed_students/data/"
TARGET = "OUTPUT Grade"
N_CLUSTERS = 8

t0 = time.time()

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Unsupervised: fit on combined feature space (train+test), since labels are only used for eval.
y_train = train[TARGET].values
y_test  = test[TARGET].values
X_train = train.drop(columns=[TARGET])
X_test  = test.drop(columns=[TARGET])
X_all   = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# All cols are low-cardinality integer codes -> stringify + one-hot (skill_stringify_low_cardinality_int_codes)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_all_oh = ohe.fit_transform(X_all.astype(str))

# Optional scaling (OHE is already 0/1 but standardize for distance methods)
scaler = StandardScaler(with_mean=False)
X_all_s = scaler.fit_transform(X_all_oh)

n_tr = len(X_train)
Xtr = X_all_s[:n_tr]
Xte = X_all_s[n_tr:]

# Small Pareto grid over clusterers + optional PCA dims (skill_multi_axis_constraint_budgets)
candidates = []
for n_comp in [None, 10, 20]:
    if n_comp is None:
        Xtr_v, Xte_v = Xtr, Xte
        tag_pca = "raw"
    else:
        pca = PCA(n_components=n_comp, random_state=0)
        Xtr_v = pca.fit_transform(np.vstack([Xtr, Xte]))
        Xte_v = Xtr_v[n_tr:]
        Xtr_v = Xtr_v[:n_tr]
        tag_pca = f"pca{n_comp}"
    for seed in [0, 1, 7, 42]:
        km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=seed)
        full = np.vstack([Xtr_v, Xte_v])
        labels_all = km.fit_predict(full)
        lt = labels_all[:n_tr]; le = labels_all[n_tr:]
        candidates.append(("kmeans", tag_pca, seed, lt, le))
    # Agglomerative on full
    for linkage in ["ward", "average"]:
        ac = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=linkage)
        full = np.vstack([Xtr_v, Xte_v])
        labels_all = ac.fit_predict(full)
        candidates.append(("agg-" + linkage, tag_pca, 0, labels_all[:n_tr], labels_all[n_tr:]))
    # GMM
    for seed in [0, 42]:
        try:
            gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=seed, n_init=5,
                                  covariance_type="diag", reg_covar=1e-3)
            full = np.vstack([Xtr_v, Xte_v])
            labels_all = gmm.fit_predict(full)
            candidates.append(("gmm-diag", tag_pca, seed, labels_all[:n_tr], labels_all[n_tr:]))
        except Exception:
            pass

# Pick the candidate with highest train rand_score (no test labels used for selection).
best = None
for tag in candidates:
    name, pca_tag, seed, lt, le = tag
    r = rand_score(y_train, lt)
    if best is None or r > best[0]:
        best = (r, name, pca_tag, seed, lt, le)

train_rand = float(best[0])
test_rand  = float(rand_score(y_test, best[5]))
elapsed    = time.time() - t0

target_met = bool(test_rand >= 0.8 and elapsed <= 600)

# Submission: predicted cluster ids on test
sub = pd.DataFrame({"id": np.arange(len(y_test)), "cluster": best[5]})
sub.to_csv("submission.csv", index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "rand_score",
    "headline_value": test_rand,
    "all_metrics": {
        "train_rand": train_rand,
        "test_rand":  test_rand,
        "training_time_s": elapsed,
        "target_met": target_met,
    },
    "skills_applied": [
        "skill_stringify_low_cardinality_int_codes.md",
        "skill_multi_axis_constraint_budgets.md",
    ],
    "interpretation_notes": (
        f"Selected {best[1]} ({best[2]}, seed={best[3]}) via train-rand selection over "
        f"KMeans/Agglomerative/GMM x raw/PCA10/PCA20. One-hot encoded all 31 int-coded "
        f"feature columns. Selection used only train labels; test labels held out for eval."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"best={best[1]} pca={best[2]} seed={best[3]} train_rand={train_rand:.4f} "
      f"test_rand={test_rand:.4f} time={elapsed:.1f}s target_met={target_met}")
