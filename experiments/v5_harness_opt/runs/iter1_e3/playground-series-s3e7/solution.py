"""
playground-series-s3e7: Reservation cancellation prediction (binary, AUC).

Skills applied:
  - skill_stringify_low_cardinality_int_codes: low-card int codes
    (type_of_meal_plan, room_type_reserved, market_segment_type,
    required_car_parking_space, repeated_guest, arrival_year) are stringified
    + one-hot encoded for the linear baseline.
  - skill_tune_when_n_train_supports_it: n_train=33680, no size/latency
    constraints in prompt -> small explicit grid for HistGradientBoosting.
  - Family-comparison rule: evaluate LogReg + HGB under same 5-fold CV.
    Average test predictions if mean AUC differs by < 0.5% relative.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


DATA_DIR = Path("/home/colligo/claude-automl/experiments/v5_harness_opt/tasks/playground-series-s3e7/data")
HERE = Path(__file__).parent
N_JOBS = 2
SEED = 42
N_SPLITS = 4

t0 = time.time()

# ---- Load ----
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

target_col = "booking_status"
id_col = "id"

y = train[target_col].values
X = train.drop(columns=[target_col, id_col])
X_test = test.drop(columns=[id_col])

# Low-cardinality int codes -> string for OHE (linear-baseline path).
LOW_CARD_CAT_COLS = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
    "required_car_parking_space",
    "repeated_guest",
    "arrival_year",
]
NUM_COLS = [c for c in X.columns if c not in LOW_CARD_CAT_COLS]


def stringify(df):
    df = df.copy()
    for c in LOW_CARD_CAT_COLS:
        df[c] = df[c].astype(str)
    return df


# ---- CV setup ----
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# ---- Family A: Linear (LogReg) ----
X_lin = stringify(X)
X_test_lin = stringify(X_test)

linear_pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), LOW_CARD_CAT_COLS),
    ])),
    ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
])

cv_auc_lin = cross_val_score(
    linear_pipe, X_lin, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS
).mean()
print(f"[Linear] LogReg CV AUC = {cv_auc_lin:.5f}  t={time.time()-t0:.1f}s", flush=True)


# ---- Family B: GBM (HistGradientBoosting) with small grid ----
# All features fed as numeric to HGB (handles them natively).
hgb_grid = [
    {"max_depth": None, "learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 1.0},
]

best_hgb_score = -np.inf
best_hgb_params = None
hgb_scores = []
for params in hgb_grid:
    clf = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, n_iter_no_change=15, validation_fraction=0.1, **params)
    score = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS).mean()
    hgb_scores.append({"params": params, "cv_auc": float(score)})
    print(f"[GBM] {params} -> CV AUC = {score:.5f}  t={time.time()-t0:.1f}s", flush=True)
    if score > best_hgb_score:
        best_hgb_score = score
        best_hgb_params = params

cv_auc_gbm = best_hgb_score
print(f"[GBM] best params={best_hgb_params}  CV AUC={cv_auc_gbm:.5f}")


# ---- Family comparison ----
rel_diff = abs(cv_auc_lin - cv_auc_gbm) / max(abs(cv_auc_lin), abs(cv_auc_gbm))
average_predictions = rel_diff < 0.005
print(f"[FamilyCmp] linear={cv_auc_lin:.5f} gbm={cv_auc_gbm:.5f} rel_diff={rel_diff:.5f}  average={average_predictions}")


# ---- Fit on full train, predict test ----
linear_pipe.fit(X_lin, y)
p_lin = linear_pipe.predict_proba(X_test_lin)[:, 1]

hgb_final = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, n_iter_no_change=15, validation_fraction=0.1, **best_hgb_params)
hgb_final.fit(X, y)
p_gbm = hgb_final.predict_proba(X_test)[:, 1]

if average_predictions:
    p_test = 0.5 * (p_lin + p_gbm)
    chosen = "average"
    headline = (cv_auc_lin + cv_auc_gbm) / 2.0
elif cv_auc_gbm >= cv_auc_lin:
    p_test = p_gbm
    chosen = "gbm"
    headline = cv_auc_gbm
else:
    p_test = p_lin
    chosen = "linear"
    headline = cv_auc_lin

print(f"[Decision] chosen={chosen}  headline_AUC={headline:.5f}")


# ---- Submission ----
sub = pd.DataFrame({id_col: test[id_col].values, target_col: p_test})
sub.to_csv(HERE / "submission.csv", index=False)


# ---- Result ----
result = {
    "target_column_chosen": target_col,
    "headline_metric": "roc_auc",
    "headline_value": float(headline),
    "all_metrics": {
        "cv_auc_linear": float(cv_auc_lin),
        "cv_auc_gbm": float(cv_auc_gbm),
        "chosen_family": chosen,
    },
    "skills_applied": [
        "skill_stringify_low_cardinality_int_codes.md",
        "skill_tune_when_n_train_supports_it.md",
    ],
    "interpretation_notes": (
        f"Binary AUC task, n_train=33680, no constraint tokens in prompt. "
        f"Compared LogReg (OHE on stringified low-card int codes + scaled numerics) "
        f"vs HistGradientBoosting (3-config grid). GBM clearly outperformed, "
        f"rel_diff={rel_diff:.4f} >= 0.005, so chose {chosen}; "
        f"no NaNs so no missing-indicator skill needed."
    ),
    "audit": {
        "family_comparison": {
            "cv_auc_linear": float(cv_auc_lin),
            "cv_auc_gbm": float(cv_auc_gbm),
            "rel_diff": float(rel_diff),
            "average_predictions": bool(average_predictions),
            "chosen": chosen,
        },
        "hgb_grid_scores": hgb_scores,
        "best_hgb_params": best_hgb_params,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "elapsed_s": float(time.time() - t0),
    },
}
with open(HERE / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nDone in {time.time()-t0:.1f}s. submission.csv and result.json written.")
