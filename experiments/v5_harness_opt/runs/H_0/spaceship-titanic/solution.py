"""Spaceship Titanic — predict `Transported` (binary classification, accuracy).

Skills applied:
  - skill_probe_group_structure_in_ids: PassengerId is `gggg_pp` (compound) and ~16%
    of train groups appear in test → use StratifiedGroupKFold keyed on `Group`.
  - skill_missing_indicator_when_missingness_signals_class: per-column NaN rates are
    ~2% (just under the 5% strict threshold) but ~24% of rows have at least one NaN
    and the cross-task evidence in the skill explicitly cites this very task with a
    +0.0098 gap from silent imputation. We add `*_missing` flags for every feature
    column with ≥1% NaN and use NA-as-category for categoricals.
  - skill_tune_when_n_train_supports_it: n_train=6954 ≫ 500, no size/latency tokens
    in the prompt, HistGradientBoosting → small explicit 4-config grid evaluated
    under the chosen group-aware CV.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/spaceship-titanic/data")
PROMPT_PATH = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/spaceship-titanic/full_prompt.txt")

TARGET = "Transported"
ID = "PassengerId"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    prompt = PROMPT_PATH.read_text()
    return train, test, prompt


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Compound-ID group structure
    out["Group"] = out[ID].str.split("_").str[0]
    out["GroupNum"] = out[ID].str.split("_").str[1].astype(int)
    # Cabin -> deck / num / side
    cabin = out["Cabin"].astype(str).str.split("/", expand=True)
    cabin.columns = ["Deck", "CabinNum", "Side"]
    cabin.loc[out["Cabin"].isna(), :] = np.nan
    out["Deck"] = cabin["Deck"]
    out["Side"] = cabin["Side"]
    out["CabinNum"] = pd.to_numeric(cabin["CabinNum"], errors="coerce")
    # Spend totals
    out["TotalSpend"] = out[SPEND_COLS].sum(axis=1, min_count=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    # Family surname (last name) — many group members share it
    out["Surname"] = out["Name"].astype(str).str.split(" ").str[-1]
    out.loc[out["Name"].isna(), "Surname"] = np.nan
    return out


def add_missingness_indicators(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Per skill_missing_indicator_when_missingness_signals_class — add binary
    indicators for any column with ≥1% NaN (relaxed from 5% because the original
    spaceship-titanic failure cited even Age/HomePlanet/CryoSleep at ~2% missing)."""
    added: list[str] = []
    exclude = {TARGET, ID, "Group", "Surname"}
    for c in train.columns:
        if c in exclude:
            continue
        if c not in test.columns:
            continue
        rate = train[c].isna().mean()
        if rate >= 0.01:
            train[f"{c}_missing"] = train[c].isna().astype(int)
            test[f"{c}_missing"] = test[c].isna().astype(int)
            added.append(c)
    return added


def probe_group_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Per skill_probe_group_structure_in_ids."""
    train_groups = set(train["Group"].astype(str))
    test_groups = set(test["Group"].astype(str))
    overlap = len(train_groups & test_groups) / max(1, len(test_groups))
    pid_compound_rate = train[ID].astype(str).str.match(r"^[A-Za-z0-9]+_[A-Za-z0-9]+$").mean()
    return {
        "group_col": "Group (PassengerId.split('_')[0])",
        "test_groups_seen_in_train_frac": float(overlap),
        "passenger_id_compound_match_frac": float(pid_compound_rate),
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
    }


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    # HistGradientBoosting handles NaN natively → no imputer needed.
    # OneHot for low-card categoricals (kept stable for HGBM).
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    train_raw, test_raw, prompt = load()

    train = feature_engineer(train_raw)
    test = feature_engineer(test_raw)

    # NA-as-category for object columns (per missing-indicator skill action)
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    for c in cat_cols:
        train[c] = train[c].astype("object").where(train[c].notna(), "__missing__")
        test[c] = test[c].astype("object").where(test[c].notna(), "__missing__")

    # Cast bool-ish columns to category strings
    for c in ["CryoSleep", "VIP"]:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    indicators = add_missingness_indicators(train, test)

    # Group-leakage probe
    leakage_audit = probe_group_leakage(train, test)
    print("group-leakage probe:", leakage_audit)

    y = train[TARGET].astype(int).values
    groups = train["Group"].values

    feat_numeric = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "CabinNum", "GroupNum", "TotalSpend", "NoSpend",
    ] + [f"{c}_missing" for c in indicators]
    feat_categorical = cat_cols
    X_train = train[feat_numeric + feat_categorical]
    X_test = test[feat_numeric + feat_categorical]

    # CV: StratifiedGroupKFold keyed on Group (skill_probe_group_structure)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    # Small explicit grid (skill_tune_when_n_train_supports_it)
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0),
        dict(max_iter=600, learning_rate=0.03, max_depth=None, max_leaf_nodes=31, l2_regularization=0.0),
        dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=63, l2_regularization=0.1),
        dict(max_iter=500, learning_rate=0.04, max_depth=8,    max_leaf_nodes=31, l2_regularization=0.0),
    ]

    results = []
    for params in grid:
        pipe = build_pipeline(feat_numeric, feat_categorical)
        pipe.set_params(
            clf__max_iter=params["max_iter"],
            clf__learning_rate=params["learning_rate"],
            clf__max_depth=params["max_depth"],
            clf__max_leaf_nodes=params["max_leaf_nodes"],
            clf__l2_regularization=params["l2_regularization"],
            clf__random_state=0,
            clf__early_stopping=False,
        )
        scores = cross_val_score(
            pipe, X_train, y, cv=cv.split(X_train, y, groups),
            scoring="accuracy", n_jobs=1,
        )
        mean, std = float(scores.mean()), float(scores.std())
        results.append((mean, std, params))
        print(f"  {params}: {mean:.4f} ± {std:.4f}")

    results.sort(key=lambda r: -r[0])
    best_mean, best_std, best_params = results[0]
    print(f"best: {best_params} → {best_mean:.4f} ± {best_std:.4f}")

    # Also report plain (non-group) CV for the gap audit
    from sklearn.model_selection import StratifiedKFold
    plain_cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe_best = build_pipeline(feat_numeric, feat_categorical)
    pipe_best.set_params(
        clf__max_iter=best_params["max_iter"],
        clf__learning_rate=best_params["learning_rate"],
        clf__max_depth=best_params["max_depth"],
        clf__max_leaf_nodes=best_params["max_leaf_nodes"],
        clf__l2_regularization=best_params["l2_regularization"],
        clf__random_state=0,
        clf__early_stopping=False,
    )
    plain_scores = cross_val_score(pipe_best, X_train, y, cv=plain_cv, scoring="accuracy", n_jobs=1)
    plain_mean = float(plain_scores.mean())
    print(f"plain stratified CV (audit): {plain_mean:.4f} (gap vs group: {plain_mean - best_mean:+.4f})")

    # Refit on full train and predict
    final = build_pipeline(feat_numeric, feat_categorical)
    final.set_params(
        clf__max_iter=best_params["max_iter"],
        clf__learning_rate=best_params["learning_rate"],
        clf__max_depth=best_params["max_depth"],
        clf__max_leaf_nodes=best_params["max_leaf_nodes"],
        clf__l2_regularization=best_params["l2_regularization"],
        clf__random_state=0,
        clf__early_stopping=False,
    )
    final.fit(X_train, y)
    pred = final.predict(X_test)

    sub = pd.DataFrame({
        ID: test_raw[ID].values,
        TARGET: pred.astype(bool),
    })
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path} ({len(sub)} rows)")

    result = {
        "target_column_chosen": TARGET,
        "headline_metric": "accuracy",
        "headline_value": best_mean,
        "all_metrics": {
            "cv_accuracy_group_mean": best_mean,
            "cv_accuracy_group_std": best_std,
            "cv_accuracy_plain_stratified_mean": plain_mean,
            "cv_optimism_gap_plain_minus_group": plain_mean - best_mean,
        },
        "skills_applied": [
            "skill_probe_group_structure_in_ids.md",
            "skill_missing_indicator_when_missingness_signals_class.md",
            "skill_tune_when_n_train_supports_it.md",
        ],
        "interpretation_notes": (
            "PassengerId is compound `gggg_pp`; ~16% of test groups appear in train, "
            "so CV uses StratifiedGroupKFold keyed on the group prefix to avoid the "
            "~0.011 optimism reported in v1. Per-column NaN rates are ~2% but ~24% of "
            "rows have at least one NaN, so we add *_missing indicators (rate>=1%) and "
            "treat NaN as its own category for categoricals. HistGradientBoosting is "
            "tuned over a 4-config grid; n_train=6954 ≫ 500 with no size/latency tokens "
            "in the prompt."
        ),
        "audit": {
            "group_leakage_probe": leakage_audit,
            "missingness_indicators_added": indicators,
            "hyperparameter_grid": [
                {"params": p, "mean": m, "std": s} for m, s, p in results
            ],
            "chosen_config": best_params,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "features_numeric": feat_numeric,
            "features_categorical": feat_categorical,
        },
    }
    result_path = HERE / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
