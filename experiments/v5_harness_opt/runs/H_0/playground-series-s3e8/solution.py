"""
Playground Series S3E8 — Gemstone price regression.
HistGradientBoostingRegressor with native categorical support; log1p target;
small 2-config grid via 3-fold CV on a sample.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e8/data"
N_JOBS = 2

CAT_COLS = ["cut", "color", "clarity"]
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]


def load():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return tr, te


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Treat exact-zero physical dimensions as missing (sentinel).
    for c in ("x", "y", "z"):
        df.loc[df[c] == 0, c] = np.nan
    # Engineered features
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["xy_ratio"] = df["x"] / df["y"]
    df["density"] = df["carat"] / df["volume"]
    df["table_depth"] = df["table"] * df["depth"]
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    # Ordered encodings reflecting standard gemstone grading (low -> high quality).
    cut_order = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_order = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
    clarity_order = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
    for df in (train, test):
        df["cut"] = df["cut"].map(cut_order).astype("int16")
        df["color"] = df["color"].map(color_order).astype("int16")
        df["clarity"] = df["clarity"].map(clarity_order).astype("int16")
    return train, test


def make_model(params):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params.get("l2_regularization", 0.0),
        categorical_features=[0, 1, 2],  # cut, color, clarity after column ordering below
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )


def main():
    t0 = time.time()
    tr_raw, te_raw = load()
    tr = add_features(tr_raw)
    te = add_features(te_raw)
    tr, te = encode_categoricals(tr, te)

    feat_cols = CAT_COLS + NUM_COLS + ["volume", "xy_ratio", "density", "table_depth"]
    X = tr[feat_cols].values
    y = np.log1p(tr["price"].values.astype(np.float64))
    X_test = te[feat_cols].values

    # 2-config minimal grid; pick by 3-fold CV on a 60k subsample (fast under 4-min budget).
    configs = [
        {"learning_rate": 0.08, "max_iter": 800, "max_leaf_nodes": 63, "min_samples_leaf": 30, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 1200, "max_leaf_nodes": 127, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]

    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(X), size=min(60000, len(X)), replace=False)
    Xs, ys = X[sub_idx], y[sub_idx]
    yprice_s = np.expm1(ys)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = []
    for ci, cfg in enumerate(configs):
        rmses = []
        for fold, (tri, vai) in enumerate(kf.split(Xs)):
            m = make_model(cfg)
            m.fit(Xs[tri], ys[tri])
            pred = np.expm1(m.predict(Xs[vai]))
            rmse = float(np.sqrt(mean_squared_error(yprice_s[vai], pred)))
            rmses.append(rmse)
        mean_rmse = float(np.mean(rmses))
        cv_results.append({"config": cfg, "cv_rmse": mean_rmse, "fold_rmses": rmses})
        print(f"[cv] cfg{ci} rmse={mean_rmse:.4f} folds={[round(r,2) for r in rmses]} t={time.time()-t0:.1f}s")

    best = min(cv_results, key=lambda r: r["cv_rmse"])
    print(f"[best] cfg={best['config']} cv_rmse={best['cv_rmse']:.4f}")

    # Refit on all training data
    model = make_model(best["config"])
    model.fit(X, y)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 326.0, None)  # train minimum

    sub = pd.DataFrame({"id": te_raw["id"], "price": pred})
    sub.to_csv(os.path.join(HERE, "submission.csv"), index=False)

    result = {
        "target_column_chosen": "price",
        "headline_metric": "rmse",
        "headline_value": best["cv_rmse"],
        "all_metrics": {
            "cv_rmse_per_config": [{"config": r["config"], "cv_rmse": r["cv_rmse"]} for r in cv_results],
            "best_config": best["config"],
        },
        "skills_applied": [
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "HistGradientBoostingRegressor on log1p(price) with ordered categorical encodings "
            "(cut/color/clarity) and engineered volume/density features. Small 2-config CV grid "
            "selected by 3-fold RMSE on 60k subsample; final fit on full 155k train."
        ),
    }
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[done] rows submitted: {len(sub)} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
