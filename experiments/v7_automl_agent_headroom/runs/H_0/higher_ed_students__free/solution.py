"""Unsupervised clustering of Higher Ed Students into 8 groups.

Score Rand Index vs OUTPUT Grade. All features are low-cardinality int codes
(ordinal/categorical questionnaire answers), so we one-hot encode and then
run KMeans on the resulting binary representation. We try a few KMeans seeds
and pick the one with best train silhouette (purely unsupervised model
selection), then evaluate Rand Index on train and test.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, silhouette_score
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/higher_ed_students/data"

TARGET = "OUTPUT Grade"
N_CLUSTERS = 8

train = pd.read_csv(f"{DATA}/train.csv")
valid = pd.read_csv(f"{DATA}/valid.csv")
test = pd.read_csv(f"{DATA}/test.csv")

y_train = train[TARGET].values
y_test = test[TARGET].values

X_train_raw = train.drop(columns=[TARGET])
X_test_raw = test.drop(columns=[TARGET])

# Stringify low-cardinality int codes before one-hot (skill_stringify_low_cardinality_int_codes).
# All columns are int codes for questionnaire answers; treat as categorical.
for c in X_train_raw.columns:
    X_train_raw[c] = X_train_raw[c].astype(str)
    X_test_raw[c] = X_test_raw[c].astype(str)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

X_train = ohe.fit_transform(X_train_raw)
X_test = ohe.transform(X_test_raw)

# Optional PCA to denoise (small-n, high-dim after one-hot).
pca = PCA(n_components=min(20, X_train.shape[1], X_train.shape[0] - 1), random_state=0)
X_train_p = pca.fit_transform(X_train)
X_test_p = pca.transform(X_test)

# Try several seeds, pick best by silhouette on train (unsupervised selection).
best = None
for seed in range(20):
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
    labels = km.fit_predict(X_train_p)
    if len(set(labels)) < 2:
        continue
    sil = silhouette_score(X_train_p, labels)
    if best is None or sil > best[0]:
        best = (sil, km, labels)

_, km, train_labels = best
test_labels = km.predict(X_test_p)

train_rand = float(rand_score(y_train, train_labels))
test_rand = float(rand_score(y_test, test_labels))

# Submission: cluster assignment per test row.
sub = pd.DataFrame({"cluster": test_labels})
sub.to_csv(os.path.join(HERE, "submission.csv"), index=False)

result = {
    "target_column_chosen": TARGET,
    "headline_metric": "rand_score",
    "headline_value": test_rand,
    "all_metrics": {"train_rand": train_rand, "test_rand": test_rand},
    "skills_applied": ["skill_stringify_low_cardinality_int_codes.md"],
    "interpretation_notes": (
        "All features are low-cardinality int questionnaire codes; stringified and "
        "one-hot encoded, then PCA(20) + KMeans(k=8). Selected seed by max train "
        "silhouette (unsupervised). 'OUTPUT Grade' used only for Rand Index scoring."
    ),
}
with open(os.path.join(HERE, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
