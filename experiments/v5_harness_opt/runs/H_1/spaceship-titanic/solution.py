"""
Spaceship Titanic — binary classification, metric = accuracy.

Skills applied:
  - skill_probe_group_structure_in_ids: parse PassengerId 'gggg_pp' for group,
    parse Cabin 'deck/num/side'; use GroupKFold over the gggg group id.
  - skill_missing_indicator_when_missingness_signals_class: NA-as-category for
    categoricals, *_missing indicators for numeric amenity columns; sentinel-free.
  - skill_tune_when_n_train_supports_it: n_train = 6954 >> 500 and no
    size/latency tokens in prompt -> small explicit config grid via CV.
  - skill_blend_two_model_families: GBM (HistGradientBoosting) + ExtraTrees,
    OOF blend weight chosen by CV accuracy (ship single model if weight in {0,1}).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get(
    "AUTOML_DATA_DIR",
    "/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/spaceship-titanic/data",
))
N_JOBS = 2
N_SPLITS = 5
RNG = 42

t0 = time.time()


# ---------- feature engineering ----------

AMENITY_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS_BASE = ["HomePlanet", "CryoSleep", "Destination", "VIP",
                 "deck", "side", "group_size_bin"]
NUM_COLS_BASE = ["Age", "num"] + AMENITY_COLS + ["total_spend", "log_total_spend",
                                                  "n_nonzero_spend",
                                                  "group_size", "cabin_num"]


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # group from PassengerId
    df["group"] = df["PassengerId"].str.split("_").str[0]
    df["pp"] = df["PassengerId"].str.split("_").str[1].astype(int)
    # Cabin -> deck/num/side
    cab = df["Cabin"].fillna("NA/-1/NA").str.split("/", expand=True)
    df["deck"] = cab[0]
    df["cabin_num"] = pd.to_numeric(cab[1], errors="coerce")
    df["side"] = cab[2]
    df["num"] = df["cabin_num"]  # alias used in NUM_COLS_BASE
    # amenities
    spend = df[AMENITY_COLS].fillna(0.0)
    df["total_spend"] = spend.sum(axis=1)
    df["log_total_spend"] = np.log1p(df["total_spend"])
    df["n_nonzero_spend"] = (spend > 0).sum(axis=1)
    # group size
    gs = df.groupby("group")["PassengerId"].transform("count")
    df["group_size"] = gs
    df["group_size_bin"] = pd.cut(gs, bins=[0, 1, 2, 4, 100],
                                  labels=["solo", "pair", "small", "big"]).astype(str)
    # CryoSleep -> if true and spent 0, consistent; missing indicators
    for c in AMENITY_COLS + ["Age", "cabin_num"]:
        df[f"{c}_missing"] = df[c].isna().astype(np.int8)
    for c in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin"]:
        df[f"{c}_missing"] = df[c].isna().astype(np.int8)
    # categorical NA -> "NA"
    for c in ["HomePlanet", "CryoSleep", "Destination", "VIP", "deck", "side"]:
        df[c] = df[c].astype(object).where(df[c].notna(), "NA").astype(str)
    # CryoSleep also infer: if all amenities sum=0 and missing, likely True
    df["cryo_inferred"] = ((df["CryoSleep"] == "True") |
                            ((df["CryoSleep"] == "NA") & (df["total_spend"] == 0))).astype(np.int8)
    return df


def build_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = featurize(train_df)
    te = featurize(test_df)
    miss_cols = [c for c in tr.columns if c.endswith("_missing")] + ["cryo_inferred"]
    feature_num = NUM_COLS_BASE + miss_cols
    feature_cat = CAT_COLS_BASE

    # fill numeric NaNs with -1 (HGBT handles NaN natively; ExtraTrees needs filled)
    for c in feature_num:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        te[c] = pd.to_numeric(te[c], errors="coerce")

    # ordinal-encode categoricals (consistent across train/test, unknown -> -1)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                          encoded_missing_value=-1)
    tr_cat = enc.fit_transform(tr[feature_cat].astype(str))
    te_cat = enc.transform(te[feature_cat].astype(str))
    tr_cat = pd.DataFrame(tr_cat, columns=feature_cat, index=tr.index)
    te_cat = pd.DataFrame(te_cat, columns=feature_cat, index=te.index)

    X_tr = pd.concat([tr[feature_num], tr_cat], axis=1)
    X_te = pd.concat([te[feature_num], te_cat], axis=1)
    y = train_df["Transported"].astype(int).values
    groups = tr["group"].values
    return X_tr, X_te, y, groups, feature_cat


# ---------- modeling ----------

def oof_predict(model_factory, X, y, groups, X_te, fit_kwargs=None,
                fillna_for_et=False):
    fit_kwargs = fit_kwargs or {}
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_te))
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        Xtr_f, Xva_f = X.iloc[tr_idx], X.iloc[va_idx]
        ytr_f = y[tr_idx]
        Xte_f = X_te
        if fillna_for_et:
            Xtr_f = Xtr_f.fillna(-1)
            Xva_f = Xva_f.fillna(-1)
            Xte_f = X_te.fillna(-1)
        model = model_factory()
        model.fit(Xtr_f, ytr_f, **fit_kwargs)
        oof[va_idx] = model.predict_proba(Xva_f)[:, 1]
        test_pred += model.predict_proba(Xte_f)[:, 1] / N_SPLITS
    return oof, test_pred


def main():
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    X, X_te, y, groups, cat_cols = build_xy(train_df, test_df)
    print(f"X train: {X.shape}, X test: {X_te.shape}, t={time.time()-t0:.1f}s")

    # --- HistGradientBoosting: small tuning grid (skill 7) ---
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    hgbt_configs = [
        dict(learning_rate=0.07, max_iter=400, max_leaf_nodes=31, min_samples_leaf=20,
             l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20,
             l2_regularization=0.0),
        dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=20,
             l2_regularization=1.0),
    ]
    hgbt_results = []
    for cfg in hgbt_configs:
        def factory(c=cfg):
            return HistGradientBoostingClassifier(
                random_state=RNG,
                categorical_features=cat_idx,
                early_stopping=False,
                **c,
            )
        oof, te_pred = oof_predict(factory, X, y, groups, X_te)
        acc = accuracy_score(y, (oof >= 0.5).astype(int))
        hgbt_results.append((acc, cfg, oof, te_pred))
        print(f"  HGBT {cfg} -> acc {acc:.4f}, t={time.time()-t0:.1f}s")

    hgbt_results.sort(key=lambda r: -r[0])
    hgbt_acc, hgbt_cfg, hgbt_oof, hgbt_te = hgbt_results[0]
    print(f"Best HGBT acc {hgbt_acc:.4f} cfg {hgbt_cfg}")

    # --- ExtraTrees blend partner (skill 10) ---
    def et_factory():
        return ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RNG,
            n_jobs=N_JOBS,
        )

    et_oof, et_te = oof_predict(et_factory, X, y, groups, X_te, fillna_for_et=True)
    et_acc = accuracy_score(y, (et_oof >= 0.5).astype(int))
    print(f"ExtraTrees acc {et_acc:.4f}, t={time.time()-t0:.1f}s")

    # weight sweep
    best_w, best_acc = 0.0, hgbt_acc
    for w in np.arange(0.0, 1.0001, 0.05):
        blend = w * hgbt_oof + (1 - w) * et_oof
        a = accuracy_score(y, (blend >= 0.5).astype(int))
        if a > best_acc:
            best_acc, best_w = a, float(w)
    print(f"Blend best w(HGBT)={best_w:.2f} -> acc {best_acc:.4f}")

    if best_w in (0.0, 1.0):
        # ship single model
        if best_w == 1.0:
            final_te = hgbt_te
            chosen = "HGBT only"
        else:
            final_te = et_te
            chosen = "ExtraTrees only"
    else:
        final_te = best_w * hgbt_te + (1 - best_w) * et_te
        chosen = f"blend HGBT*{best_w:.2f} + ET*{1-best_w:.2f}"
    print(f"Final: {chosen}")

    pred = (final_te >= 0.5)

    sub = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": pred.astype(bool),
    })
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"Wrote submission.csv ({len(sub)} rows), t={time.time()-t0:.1f}s")

    result = {
        "target_column_chosen": "Transported",
        "headline_metric": "accuracy",
        "headline_value": float(best_acc),
        "all_metrics": {
            "cv_accuracy_best_hgbt": float(hgbt_acc),
            "cv_accuracy_extratrees": float(et_acc),
            "cv_accuracy_blend": float(best_acc),
            "blend_weight_hgbt": best_w,
            "n_train": int(len(y)),
            "n_test": int(len(test_df)),
            "n_splits": N_SPLITS,
            "best_hgbt_cfg": hgbt_cfg,
            "final_model": chosen,
        },
        "skills_applied": [
            "skill_probe_group_structure_in_ids.md",
            "skill_missing_indicator_when_missingness_signals_class.md",
            "skill_tune_when_n_train_supports_it.md",
            "skill_blend_two_model_families.md",
        ],
        "skills_evaluated_not_applied": [
            "skill_train_on_listed_auxiliary_files.md: no aux files in prompt.",
            "skill_verify_split_claim_before_holdout.md: prompt makes no structural split claim.",
            "skill_stringify_low_cardinality_int_codes.md: no int-coded categoricals; we use HGBT/tree models that handle features natively.",
            "skill_multi_axis_constraint_budgets.md: prompt names no production/latency/size constraints.",
            "skill_tune_decision_threshold_for_f1.md: metric is accuracy, not F1/F-beta/MCC; 0.5 is optimal for accuracy on calibrated probs.",
            "skill_log_target_for_skewed_nonnegative_regression.md: classification, not regression.",
        ],
        "interpretation_notes": (
            "Parsed PassengerId group + Cabin deck/num/side; group-aware GroupKFold "
            "(n=5) since groups span train/test. Engineered total_spend, log_total_spend, "
            "n_nonzero_spend, group_size, and missingness indicators. Tuned 3 HGBT "
            "configs with native categorical support, then blended with ExtraTrees by "
            "OOF weight search; threshold left at 0.5 because metric is accuracy."
        ),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(HERE / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result.json, total t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
