"""Spaceship Titanic — target=Transported, addressing Challenger audit.

Design choices (each tied to an audit item):

  (1) Target column: `Transported`. Justification = "only column in train.csv
      but not test.csv" (set(train.cols) - set(test.cols) == {'Transported'}).
      Not relying on sample_submission.csv (Gap A — IDs don't match test).

  (2) Submission keyed on test.csv's 1,739 PassengerIds, header
      `PassengerId,Transported`, boolean values. (Spirit-of-spec #2, Gap F)

  (3) CV split is group-aware via `gggg` prefix of PassengerId
      (GroupKFold) — required by Gap B. We report both per-row and
      group-aware CV scores so the leakage gap is visible.

  (4) No groupmate-label leakage features are used. We use group-aware
      aggregates of *features only* (not labels), and we explicitly do NOT
      build groupmate_transported_mean. (Spirit-of-spec #4)

  (5) Missingness policy is declared per column:
        - Categoricals (HomePlanet, CryoSleep, Destination, VIP, Deck, Side):
          NA-as-category ('__NA__'), one-hot encoded.
        - Spend columns: NA -> 0 (zero-spend is the dominant mode and
          consistent with CryoSleep invariant). Plus log1p transform + a
          per-column `was_missing` flag.
        - Age: median imputed + `Age_missing` flag + child indicator.
      (Gap C)

  (6) CryoSleep⇒zero-spend invariant is exploited: when CryoSleep is NA
      and all five spend columns are 0 (or NA->0) we set CryoSleep=True;
      when CryoSleep is NA and any spend > 0 we set CryoSleep=False.
      (Gap D)

  (7) Cabin is split into Deck / Num / Side; raw Cabin string is dropped.
      Deck=='T' (5 train rows) is folded into a small '__rare__' bucket to
      survive group-aware CV. (Gap E)

  Other defended items:
    - Gap G: `Child` flag (Age<=12) preserved as feature.
    - Gap H: no group-majority-label feature (only 46% homogeneous groups).
    - Group-size feature derived from PassengerId is fine (not label-leaking).

Model: HistGradientBoostingClassifier (sklearn, no GPU, robust to mixed
feature distributions). Final fit on full train, predicts test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, KFold

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
SEED = 42

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS_RAW = ["HomePlanet", "CryoSleep", "Destination", "VIP"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    # --- Audit item #1: confirm target structurally ---
    diff = set(train.columns) - set(test.columns)
    assert diff == {"Transported"}, (
        f"Target structurally inferred from schema diff; expected "
        f"{{'Transported'}}, got {diff}"
    )
    return train, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Group ID from PassengerId (gggg_pp). NOT label-derived.
    out["GroupId"] = out["PassengerId"].str.split("_").str[0]
    out["PaxInGroup"] = out["PassengerId"].str.split("_").str[1].astype(int)
    out["GroupSize"] = out.groupby("GroupId")["PassengerId"].transform("count")

    # Cabin -> Deck / Num / Side  (Gap E)
    cabin_split = out["Cabin"].str.split("/", expand=True)
    out["Deck"] = cabin_split[0]
    out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["Side"] = cabin_split[2]

    # Fold rare Deck 'T' (5 train rows) into '__rare__' to survive CV.
    out.loc[out["Deck"] == "T", "Deck"] = "__rare__"

    # --- CryoSleep⇒zero-spend invariant (Gap D) ---
    # Fill spend NA with 0 *only* when CryoSleep==True (consistent with invariant);
    # but spend NA otherwise is also overwhelmingly 0 (no-spend is mode), so
    # do NA->0 globally with a was_missing flag.
    for c in SPEND_COLS:
        out[f"{c}_isna"] = out[c].isna().astype(np.int8)
        out[c] = out[c].fillna(0.0)
        out[f"{c}_log1p"] = np.log1p(out[c])

    out["TotalSpend"] = out[SPEND_COLS].sum(axis=1)
    out["TotalSpend_log1p"] = np.log1p(out["TotalSpend"])
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(np.int8)

    # CryoSleep imputation from invariant:
    cs_na = out["CryoSleep"].isna()
    out.loc[cs_na & (out["TotalSpend"] == 0), "CryoSleep"] = True
    out.loc[cs_na & (out["TotalSpend"] > 0), "CryoSleep"] = False
    # any residual NA (shouldn't happen, but be safe) -> NA-as-category
    out["CryoSleep"] = out["CryoSleep"].astype("object").where(out["CryoSleep"].notna(), "__NA__")

    # Age: median impute + flag + Child flag (Gap G)
    out["Age_isna"] = out["Age"].isna().astype(np.int8)
    out["Child"] = (out["Age"].fillna(99) <= 12).astype(np.int8)
    # median is computed from combined train+test at the top level

    # VIP / HomePlanet / Destination: NA-as-category
    for c in ["HomePlanet", "Destination", "VIP"]:
        out[c] = out[c].astype("object").where(out[c].notna(), "__NA__")

    out["Deck"] = out["Deck"].astype("object").where(out["Deck"].notna(), "__NA__")
    out["Side"] = out["Side"].astype("object").where(out["Side"].notna(), "__NA__")
    out["CabinNum_isna"] = out["CabinNum"].isna().astype(np.int8)
    # CabinNum: keep as numeric; impute -1 sentinel (tree splits handle it)
    out["CabinNum"] = out["CabinNum"].fillna(-1)

    return out


def assemble_matrix(
    train_fe: pd.DataFrame, test_fe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode categoricals fit on combined index space."""
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    num_cols = (
        ["Age", "GroupSize", "PaxInGroup", "CabinNum", "TotalSpend", "TotalSpend_log1p", "NoSpend",
         "Age_isna", "CabinNum_isna", "Child"]
        + SPEND_COLS
        + [f"{c}_log1p" for c in SPEND_COLS]
        + [f"{c}_isna" for c in SPEND_COLS]
    )

    # Age median impute jointly (consistent train+test)
    combined_age = pd.concat([train_fe["Age"], test_fe["Age"]])
    age_med = float(combined_age.median())
    train_fe = train_fe.copy()
    test_fe = test_fe.copy()
    train_fe["Age"] = train_fe["Age"].fillna(age_med)
    test_fe["Age"] = test_fe["Age"].fillna(age_med)

    combined = pd.concat([train_fe[cat_cols + num_cols].assign(_src="tr"),
                          test_fe[cat_cols + num_cols].assign(_src="te")], axis=0)
    dummies = pd.get_dummies(combined[cat_cols].astype(str), prefix=cat_cols, dummy_na=False)
    X_full = pd.concat([dummies.reset_index(drop=True),
                        combined[num_cols].reset_index(drop=True)], axis=1)
    src = combined["_src"].reset_index(drop=True).values
    X_train = X_full.iloc[src == "tr"].reset_index(drop=True)
    X_test = X_full.iloc[src == "te"].reset_index(drop=True)
    feature_names = list(X_full.columns)
    return X_train, X_test, feature_names


def _fit_eval(X, y, tr, va) -> tuple[float, float]:
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=20, random_state=SEED,
    )
    clf.fit(X.iloc[tr], y[tr])
    p = clf.predict_proba(X.iloc[va])[:, 1]
    return (
        float(accuracy_score(y[va], p >= 0.5)),
        float(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6))),
    )


def cv_eval(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict:
    """Report BOTH per-row KFold and group-aware GroupKFold accuracy (Gap B).

    3 folds (not 5) to keep wall time reasonable under a loaded machine.
    """
    results = {}

    # Per-row KFold (overestimates honest accuracy due to group leakage)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    accs, lls = [], []
    for tr, va in kf.split(X):
        a, l = _fit_eval(X, y, tr, va)
        accs.append(a); lls.append(l)
        print(f"  [per-row fold] acc={a:.4f} ll={l:.4f}", flush=True)
    results["cv_per_row_accuracy_mean"] = float(np.mean(accs))
    results["cv_per_row_accuracy_std"] = float(np.std(accs))
    results["cv_per_row_logloss_mean"] = float(np.mean(lls))
    results["cv_per_row_n_splits"] = 3

    # Group-aware GroupKFold (honest)
    gkf = GroupKFold(n_splits=3)
    accs, lls = [], []
    for tr, va in gkf.split(X, y, groups=groups):
        a, l = _fit_eval(X, y, tr, va)
        accs.append(a); lls.append(l)
        print(f"  [group   fold] acc={a:.4f} ll={l:.4f}", flush=True)
    results["cv_group_accuracy_mean"] = float(np.mean(accs))
    results["cv_group_accuracy_std"] = float(np.std(accs))
    results["cv_group_logloss_mean"] = float(np.mean(lls))
    results["cv_group_n_splits"] = 3
    return results


def main():
    train, test = load()

    # Justification trail
    schema_diff = sorted(set(train.columns) - set(test.columns))

    train_fe = build_features(train)
    test_fe = build_features(test)

    X_train, X_test, feat_names = assemble_matrix(
        train_fe.drop(columns=["Transported"]), test_fe
    )
    y_train = train_fe["Transported"].astype(bool).to_numpy()
    groups = train_fe["GroupId"].to_numpy()

    cv = cv_eval(X_train, y_train, groups)
    print("[cv]", json.dumps(cv, indent=2))

    # Final fit on all train, predict test
    final = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.07, max_leaf_nodes=31,
        min_samples_leaf=20, random_state=SEED,
    )
    final.fit(X_train, y_train)
    test_pred_proba = final.predict_proba(X_test)[:, 1]
    test_pred = test_pred_proba >= 0.5

    # --- Submission keyed on test.csv (Spirit-of-spec #2, Gap F) ---
    sub = pd.DataFrame({
        "PassengerId": test["PassengerId"].to_numpy(),
        "Transported": test_pred.astype(bool),
    })
    assert len(sub) == len(test), "submission row count must match test.csv"
    assert sub["PassengerId"].is_unique
    sub_path = HERE / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[submission] wrote {sub_path}  rows={len(sub)}  "
          f"true_rate={float(sub.Transported.mean()):.4f}")

    # Sanity vs sample_submission (Gap A) — confirm we did NOT use it.
    try:
        ss = pd.read_csv(DATA / "sample_submission.csv")
        overlap = len(set(ss["PassengerId"]) & set(sub["PassengerId"]))
        ss_check = {
            "sample_submission_rows": int(len(ss)),
            "passenger_id_overlap_with_test": int(overlap),
            "submission_rows": int(len(sub)),
        }
    except Exception as e:
        ss_check = {"error": str(e)}

    # Headline metric: honest group-aware CV accuracy.
    headline_metric = "cv_group_accuracy"
    headline_value = cv["cv_group_accuracy_mean"]

    result = {
        "target_column_chosen": "Transported",
        "target_column_justification": (
            "Transported is the only column present in train.csv but absent "
            "from test.csv (set(train.columns) - set(test.columns) == "
            f"{schema_diff}). This structural schema diff — not the "
            "sample_submission header (which is unreliable; see Gap A) — is "
            "the load-bearing reason. Other booleans (CryoSleep, VIP) and "
            "categoricals (HomePlanet, Destination, Cabin) are all present "
            "in test, so they cannot be 'the missing column'."
        ),
        "headline_metric": headline_metric,
        "headline_value": float(headline_value),
        "all_metrics": {
            **cv,
            "train_class_balance_true": float(y_train.mean()),
            "test_pred_class_balance_true": float(sub["Transported"].mean()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(X_train.shape[1]),
        },
        "audit_items_addressed": [
            "Target=Transported justified by schema diff, not sample_submission (Section 1, Gap A)",
            "Submission keyed on test.csv 1,739 IDs with header PassengerId,Transported (Gap F, Spirit #2)",
            "Group-aware CV via GroupKFold on PassengerId prefix; both per-row and group CV reported (Gap B, Spirit #3)",
            "No groupmate-label leakage features built (Spirit #4) — only label-free group features (GroupSize, PaxInGroup)",
            "Per-column missingness policy: NA-as-category for HomePlanet/CryoSleep/Destination/VIP/Deck/Side; NA->0 + was_missing flag for spend; median + Age_isna flag for Age (Gap C, Spirit #5)",
            "CryoSleep⇒zero-spend invariant used to impute CryoSleep when NA (Gap D, Spirit #6)",
            "Cabin split into Deck/Num/Side; Deck=='T' (5 rows) folded into '__rare__' bucket (Gap E, Spirit #7)",
            "Child flag (Age<=12) preserved (Gap G)",
            "No group-majority-label baseline used (Gap H — only 46% homogeneous)",
        ],
        "interpretation_notes": (
            "Headline value is group-aware 3-fold CV accuracy "
            f"({cv['cv_group_accuracy_mean']:.4f} +- "
            f"{cv['cv_group_accuracy_std']:.4f}). Per-row 3-fold CV gives "
            f"{cv['cv_per_row_accuracy_mean']:.4f}; the gap reflects "
            "group-level leakage flagged in Gap B (40.8% of test rows share "
            "a group with train). We report the group-aware number as the "
            "honest generalization estimate. Submission has "
            f"{int(len(sub))} rows matching test.csv exactly, with predicted "
            f"True-rate {float(sub.Transported.mean()):.3f} (train True-rate "
            f"{float(y_train.mean()):.3f}). sample_submission.csv was "
            f"inspected only to confirm non-overlap: {ss_check}."
        ),
    }

    res_path = HERE / "result.json"
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] wrote {res_path}")
    print(f"[headline] {headline_metric} = {headline_value:.4f}")


if __name__ == "__main__":
    main()
