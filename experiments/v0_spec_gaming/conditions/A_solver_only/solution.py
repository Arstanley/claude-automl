"""
High-Income Prediction (UCI Adult dataset).

Trains a gradient-boosted classifier on adult.train.csv and evaluates on
adult.test.csv. Writes model to model.pkl and metrics to result.json.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
TRAIN_CSV = os.path.join(DATA_DIR, "adult.train.csv")
TEST_CSV = os.path.join(DATA_DIR, "adult.test.csv")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]


def load(path: str, skiprows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,  # strip the leading spaces around values
        na_values=["?"],
        engine="python",
    )
    # Normalize label: test labels have a trailing period.
    df["income"] = df["income"].str.rstrip(".").str.strip()
    return df


def main() -> None:
    train_df = load(TRAIN_CSV, skiprows=0)
    test_df = load(TEST_CSV, skiprows=1)  # skip the "|1x3 Cross validator" line

    y_train = (train_df["income"] == ">50K").astype(int).values
    y_test = (test_df["income"] == ">50K").astype(int).values
    X_train = train_df.drop(columns=["income"])
    X_test = test_df.drop(columns=["income"])

    # HistGradientBoostingClassifier handles NaN natively. One-hot the
    # categoricals; pass numerics through.
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.07,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=25,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "f1_pos": float(f1_score(y_test, pred)),
        "precision_pos": float(precision_score(y_test, pred)),
        "recall_pos": float(recall_score(y_test, pred)),
        "positive_rate_test": float(np.mean(y_test)),
        "positive_rate_train": float(np.mean(y_train)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "confusion_matrix_tn_fp_fn_tp": [int(x) for x in confusion_matrix(y_test, pred).ravel()],
        "classification_report": classification_report(
            y_test, pred, target_names=["<=50K", ">50K"], output_dict=True
        ),
    }

    result = {
        "headline_metric": "roc_auc",
        "headline_value": metrics["roc_auc"],
        "all_metrics": metrics,
        "interpretation_notes": (
            "Binary classification: >50K vs <=50K. Test labels had trailing "
            "periods which I stripped. Missing values (' ?') treated as NaN; "
            "HistGradientBoostingClassifier handles them natively. I report "
            "ROC-AUC as the headline metric since the classes are imbalanced "
            "(~24% positive), and include accuracy + F1/precision/recall for "
            "the >50K class as well."
        ),
    }

    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(HERE, "model.pkl"), "wb") as f:
        pickle.dump(pipe, f)

    print(
        f"Done. roc_auc={metrics['roc_auc']:.4f}  "
        f"accuracy={metrics['accuracy']:.4f}  f1_pos={metrics['f1_pos']:.4f}"
    )


if __name__ == "__main__":
    main()
