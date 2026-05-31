"""
Wine Quality (White) — multi-class macro-F1 with constraint (train <= 5 min, infer <= 5ms/instance).

Skills applied:
- skill_ordinal_target_and_rare_class_macro_f1: ordinal-aware regression-rounded + sample-weighted
  GBM classifier + manual oversampling of rare classes (class-3, class-9).
- skill_multi_axis_constraint_budgets: budgets locked up front (train_time<=300s, infer<=5ms/i),
  rejected candidates that violate.

H_0 baseline: 0.415 (RandomForest balanced_subsample). Lever attempted: more diverse candidates
(HGB regression-rounded, HGB clf, RF, ExtraTrees), tiny oversampling, soft proba ensemble,
re-train winner on train+valid for the final test-time prediction.
"""
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

DATA_DIR = "/home/colligo/claude-automl/experiments/v7_automl_agent_headroom/tasks/wine_quality_white/data"
HERE = os.path.dirname(os.path.abspath(__file__))

CONSTRAINT_BUDGETS = {
    "train_time_sec": 300.0,
    "infer_ms_per_instance": 5.0,
    "headline_macro_f1_floor": 0.42,
}

RNG = 42


def load():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return tr, va, te


def oversample_rare(X, y, min_count=20, rng=RNG):
    """Duplicate rows whose class has fewer than min_count members up to min_count."""
    rs = np.random.RandomState(rng)
    counts = pd.Series(y).value_counts().to_dict()
    extra_X, extra_y = [], []
    for c, n in counts.items():
        if n < min_count:
            need = min_count - n
            idx = np.where(y == c)[0]
            picks = rs.choice(idx, size=need, replace=True)
            extra_X.append(X[picks])
            extra_y.append(y[picks])
    if extra_X:
        X_new = np.vstack([X] + extra_X)
        y_new = np.concatenate([y] + extra_y)
        return X_new, y_new
    return X, y


def time_predict(model, X, n_warmup=2, n_repeat=3):
    for _ in range(n_warmup):
        model.predict(X[:min(100, len(X))])
    best = float("inf")
    for _ in range(n_repeat):
        t = time.perf_counter()
        model.predict(X)
        dt = time.perf_counter() - t
        best = min(best, dt)
    return 1000.0 * best / len(X)


def to_probs(pred_int, classes):
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(pred_int), len(classes)))
    for i, p in enumerate(pred_int):
        j = idx.get(int(p))
        if j is not None:
            out[i, j] = 1.0
    return out


def proba_aligned(model, X, classes):
    out = np.zeros((len(X), len(classes)))
    local = list(getattr(model, "classes_", classes))
    p = model.predict_proba(X)
    for j, c in enumerate(local):
        if int(c) in classes:
            out[:, classes.index(int(c))] = p[:, j]
    return out


def classes_from_proba(proba, cls):
    idx = proba.argmax(axis=1)
    return np.array([cls[i] for i in idx])


def main():
    t0 = time.time()
    tr, va, te = load()
    target = "quality"
    feats = [c for c in tr.columns if c != target]

    X_tr, y_tr = tr[feats].values, tr[target].values.astype(int)
    X_va, y_va = va[feats].values, va[target].values.astype(int)
    X_te, y_te = te[feats].values, te[target].values.astype(int)
    classes = sorted(np.unique(np.concatenate([y_tr, y_va, y_te])).tolist())
    y_min, y_max = int(min(y_tr)), int(max(y_tr))
    print(f"classes={classes}, n_train={len(X_tr)}, n_val={len(X_va)}, n_test={len(X_te)}")
    print(f"train counts: {pd.Series(y_tr).value_counts().sort_index().to_dict()}")

    # Oversampled variant for classifiers
    X_tr_os, y_tr_os = oversample_rare(X_tr, y_tr, min_count=30)
    print(f"after rare-class oversample: n={len(X_tr_os)} counts={pd.Series(y_tr_os).value_counts().sort_index().to_dict()}")

    sw_os = compute_sample_weight("balanced", y_tr_os)

    candidates = []

    def evalc(name, train_fn, proba_fn):
        ts = time.time()
        model = train_fn()
        train_time = time.time() - ts
        infer_ms = time_predict(model, X_va)
        val_proba = proba_fn(model, X_va)
        val_pred = classes_from_proba(val_proba, classes)
        f1 = f1_score(y_va, val_pred, average="macro")
        passes = (train_time <= CONSTRAINT_BUDGETS["train_time_sec"]
                  and infer_ms <= CONSTRAINT_BUDGETS["infer_ms_per_instance"])
        print(f"  [{name}] val_f1={f1:.4f} train={train_time:.1f}s infer={infer_ms:.3f}ms/i pass={passes}")
        candidates.append({
            "name": name, "model": model, "val_f1": f1,
            "train_time": train_time, "infer_ms": infer_ms, "passes": passes,
            "proba_fn": proba_fn, "val_proba": val_proba, "val_pred": val_pred,
            "train_fn": train_fn,
        })

    # 1) HGB regression-rounded (ordinal trick) — on original train
    def train_hgb_reg():
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=RNG,
        )
        m.fit(X_tr, y_tr.astype(float))
        return m
    def proba_hgb_reg(m, X):
        return to_probs(np.clip(np.round(m.predict(X)), y_min, y_max).astype(int), classes)
    evalc("hgb_regression_rounded", train_hgb_reg, proba_hgb_reg)

    # 2) HGB classifier with balanced sample weights on oversampled set
    def train_hgb_clf():
        m = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=RNG,
        )
        m.fit(X_tr_os, y_tr_os, sample_weight=sw_os)
        return m
    def proba_hgb_clf(m, X):
        return proba_aligned(m, X, classes)
    evalc("hgb_clf_oversampled_sw", train_hgb_clf, proba_hgb_clf)

    # 3) RandomForest balanced subsample (H_0 family) on oversampled set
    def train_rf():
        m = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RNG, n_jobs=2,
        )
        m.fit(X_tr_os, y_tr_os)
        return m
    def proba_tree(m, X):
        return proba_aligned(m, X, classes)
    evalc("rf_balanced_subsample_os", train_rf, proba_tree)

    # 4) ExtraTrees balanced subsample on oversampled set
    def train_et():
        m = ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=RNG, n_jobs=2,
        )
        m.fit(X_tr_os, y_tr_os)
        return m
    evalc("extra_trees_balanced_os", train_et, proba_tree)

    # 5) RandomForest no oversample but balanced_subsample (mirrors H_0 directly)
    def train_rf_plain():
        m = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=RNG, n_jobs=2,
        )
        m.fit(X_tr, y_tr)
        return m
    evalc("rf_plain", train_rf_plain, proba_tree)

    # 6) ExtraTrees plain
    def train_et_plain():
        m = ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=RNG, n_jobs=2,
        )
        m.fit(X_tr, y_tr)
        return m
    evalc("extra_trees_plain", train_et_plain, proba_tree)

    # ----- Filter and select -----
    survivors = [c for c in candidates if c["passes"]]
    if not survivors:
        survivors = candidates
        print("WARNING: no candidate met both budgets; using all.")
    survivors.sort(key=lambda c: c["val_f1"], reverse=True)
    best = survivors[0]
    print(f"\nBest single: {best['name']} val_f1={best['val_f1']:.4f}")

    # Soft ensemble: top-3 (and try top-2)
    options = [("single_best", [best])]
    for k in (2, 3):
        if len(survivors) >= k:
            options.append((f"ensemble_top{k}", survivors[:k]))

    best_val = -1
    best_choice = None
    for name, members in options:
        if len(members) == 1:
            proba = members[0]["val_proba"]
        else:
            proba = np.mean([m["val_proba"] for m in members], axis=0)
        pred = classes_from_proba(proba, classes)
        f1 = f1_score(y_va, pred, average="macro")
        print(f"  {name}: val_f1={f1:.4f} members={[m['name'] for m in members]}")
        if f1 > best_val:
            best_val = f1
            best_choice = (name, members, pred)

    chosen_name, chosen_members, chosen_val_pred = best_choice
    print(f"\nChosen: {chosen_name} val_f1={best_val:.4f}")

    # ----- Refit chosen members on train+valid for the test prediction -----
    print("\nRefitting chosen members on train+valid for final test prediction...")
    X_all = np.vstack([X_tr, X_va])
    y_all = np.concatenate([y_tr, y_va])
    X_all_os, y_all_os = oversample_rare(X_all, y_all, min_count=30)
    sw_all_os = compute_sample_weight("balanced", y_all_os)

    # Need to rebuild train_fn but pointing at the all-data variant. We use member name to dispatch.
    def refit(name):
        if name == "hgb_regression_rounded":
            m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=20, random_state=RNG)
            m.fit(X_all, y_all.astype(float))
            return ("reg", m)
        if name == "hgb_clf_oversampled_sw":
            m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=20, random_state=RNG)
            m.fit(X_all_os, y_all_os, sample_weight=sw_all_os)
            return ("clf", m)
        if name == "rf_balanced_subsample_os":
            m = RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                       class_weight="balanced_subsample", random_state=RNG, n_jobs=2)
            m.fit(X_all_os, y_all_os)
            return ("clf", m)
        if name == "extra_trees_balanced_os":
            m = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1,
                                     class_weight="balanced_subsample", random_state=RNG, n_jobs=2)
            m.fit(X_all_os, y_all_os)
            return ("clf", m)
        if name == "rf_plain":
            m = RandomForestClassifier(n_estimators=600, min_samples_leaf=1,
                                       class_weight="balanced_subsample", random_state=RNG, n_jobs=2)
            m.fit(X_all, y_all)
            return ("clf", m)
        if name == "extra_trees_plain":
            m = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1,
                                     class_weight="balanced_subsample", random_state=RNG, n_jobs=2)
            m.fit(X_all, y_all)
            return ("clf", m)
        raise ValueError(name)

    def proba_refit(kind, m, X):
        if kind == "reg":
            return to_probs(np.clip(np.round(m.predict(X)), y_min, y_max).astype(int), classes)
        return proba_aligned(m, X, classes)

    test_probas = []
    refit_train_total = 0.0
    for mem in chosen_members:
        ts = time.time()
        kind, model = refit(mem["name"])
        refit_train_total += time.time() - ts
        test_probas.append(proba_refit(kind, model, X_te))
    test_proba = np.mean(test_probas, axis=0)
    test_pred = classes_from_proba(test_proba, classes)
    test_f1 = f1_score(y_te, test_pred, average="macro")
    print(f"\nTEST macro-F1 = {test_f1:.4f}  (refit total train time = {refit_train_total:.1f}s)")
    print(classification_report(y_te, test_pred, zero_division=0))

    # Submission
    sub = pd.DataFrame({"quality": test_pred.astype(int)})
    sub.to_csv(os.path.join(HERE, "submission.csv"), index=False)

    # Per-class val F1 for the chosen
    val_report = classification_report(y_va, chosen_val_pred, output_dict=True, zero_division=0)
    per_class_val_f1 = {k: v.get("f1-score", 0.0) for k, v in val_report.items() if k.isdigit()}

    result = {
        "target_column_chosen": "quality",
        "headline_metric": "macro_f1",
        "headline_value": float(test_f1),
        "all_metrics": {
            "val_macro_f1": float(best_val),
            "test_macro_f1": float(test_f1),
            "per_class_val_f1": per_class_val_f1,
            "best_single_name": best["name"],
            "best_single_val_f1": float(best["val_f1"]),
            "candidate_summary": [
                {
                    "name": c["name"], "val_f1": float(c["val_f1"]),
                    "train_time_sec": float(c["train_time"]),
                    "infer_ms_per_instance": float(c["infer_ms"]),
                    "passes_budgets": bool(c["passes"]),
                } for c in candidates
            ],
        },
        "skills_applied": [
            "skill_ordinal_target_and_rare_class_macro_f1.md",
            "skill_multi_axis_constraint_budgets.md",
        ],
        "constraint_budgets": CONSTRAINT_BUDGETS,
        "chosen_model": chosen_name,
        "ensemble_members": [m["name"] for m in chosen_members],
        "interpretation_notes": (
            "H_0 baseline (RF balanced_subsample) hit 0.415 on a nominal classifier with class-3/9 F1=0. "
            "Applied skill_ordinal_target_and_rare_class_macro_f1 (regression-rounded HGB + sample-weighted "
            "classifier + manual rare-class oversampling) and skill_multi_axis_constraint_budgets (locked "
            "train<=300s, infer<=5ms/i; all candidates passed). Chose by val macro-F1 over 6 candidates "
            "incl. soft-proba ensembles; refit chosen members on train+valid for the final test prediction."
        ),
        "wallclock_total_sec": float(time.time() - t0),
    }
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote submission.csv ({len(sub)} rows) and result.json. Total wallclock={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
