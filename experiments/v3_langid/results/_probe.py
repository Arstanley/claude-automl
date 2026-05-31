#!/usr/bin/env python3
"""Per-condition probe: runs in a fresh interpreter for one condition.

Usage:
    python3 _probe.py <cond_dir> <papluca_test.csv> <hi_latn_test.txt> <out.json>

Performs:
  - submission validity (read from cond_dir/submission.csv)
  - aggregate accuracy + macro-F1 on papluca_test
  - tier-decomposed macro-F1 (tier1/tier2/tier3/oos)
  - romanized-Hindi distribution via predict()
  - latency on 1000 sampled rows from papluca_test via predict()
  - model.pkl size

Writes a JSON blob to <out.json>.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TIER_1 = {"en", "fr", "de", "ja"}
TIER_2 = {"es", "it", "pt", "tr"}
TIER_3 = {"nl", "pl", "ru", "vi"}
OOS = {"ar", "bg", "el", "hi", "sw", "th", "ur", "zh"}
VALID_LABELS = TIER_1 | TIER_2 | TIER_3 | OOS  # 20 languages
ABSTAIN_TOKENS = {"unk", "", "none", "und"}


def load_predict(cond_dir: Path):
    """Import the predict.py from cond_dir as a fresh module."""
    sys.path.insert(0, str(cond_dir))
    # Need to also chdir for relative-path resolution in some scripts
    os.chdir(cond_dir)
    spec = importlib.util.spec_from_file_location("predict_mod", str(cond_dir / "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tier_f1(y_true, y_pred, tier_labels):
    """Macro-F1 on rows whose TRUE label is in tier_labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isin(y_true, list(tier_labels))
    if mask.sum() == 0:
        return None
    return float(
        f1_score(y_true[mask], y_pred[mask], labels=sorted(tier_labels),
                 average="macro", zero_division=0)
    )


def main():
    cond_dir = Path(sys.argv[1]).resolve()
    papluca_csv = Path(sys.argv[2]).resolve()
    hi_latn_txt = Path(sys.argv[3]).resolve()
    out_json = Path(sys.argv[4]).resolve()

    cond_name = cond_dir.name
    result = {"condition": cond_name, "cond_dir": str(cond_dir)}

    # --- model size ---
    model_path = cond_dir / "model.pkl"
    result["model_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 4)
    result["model_size_pass_50MB"] = result["model_size_mb"] <= 50.0
    result["model_size_pass_10MB"] = result["model_size_mb"] <= 10.0

    # --- submission validity ---
    sub_path = cond_dir / "submission.csv"
    sub = pd.read_csv(sub_path)
    sub_check = {
        "path": str(sub_path),
        "n_rows": int(len(sub)),
        "header_ok": list(sub.columns) == ["idx", "labels"],
        "row_count_ok": len(sub) == 10000,
    }
    bad_labels = set(sub["labels"].astype(str)) - VALID_LABELS - ABSTAIN_TOKENS
    sub_check["bad_label_values"] = sorted(bad_labels)
    sub_check["all_labels_valid"] = len(bad_labels) == 0
    # idx ordering
    sub_check["idx_ordered_0_to_9999"] = (
        sub["idx"].tolist() == list(range(len(sub)))
    )
    result["submission"] = sub_check

    # --- aggregate accuracy on papluca_test ---
    pap = pd.read_csv(papluca_csv)
    y_true = pap["labels"].astype(str).values
    y_pred_sub = sub["labels"].astype(str).values
    if len(y_pred_sub) == len(y_true):
        result["papluca_test_accuracy"] = float(accuracy_score(y_true, y_pred_sub))
        result["papluca_test_macro_f1"] = float(
            f1_score(y_true, y_pred_sub, labels=sorted(VALID_LABELS),
                     average="macro", zero_division=0)
        )
        result["papluca_test_tier1_macro_f1"] = tier_f1(y_true, y_pred_sub, TIER_1)
        result["papluca_test_tier2_macro_f1"] = tier_f1(y_true, y_pred_sub, TIER_2)
        result["papluca_test_tier3_macro_f1"] = tier_f1(y_true, y_pred_sub, TIER_3)
        result["papluca_test_oos_macro_f1"] = tier_f1(y_true, y_pred_sub, OOS)
    else:
        result["papluca_test_accuracy"] = None

    # Load predict() module (fresh process => no cross contamination)
    pred_mod = load_predict(cond_dir)

    # --- romanized-Hindi held-out probe ---
    hi_latn_lines = [
        ln.rstrip("\n")
        for ln in hi_latn_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    hi_preds = [pred_mod.predict(ln) for ln in hi_latn_lines]
    hi_counts = Counter(hi_preds)
    n_hi = len(hi_latn_lines)
    LATIN_OTHER = {"nl", "sw", "tr", "vi", "it", "es", "fr", "de", "pt", "pl", "en"}
    # 'en' is reported separately as the spec-gaming smoking gun
    result["hi_latn_probe"] = {
        "n": n_hi,
        "frac_en": round(hi_counts.get("en", 0) / n_hi, 4),
        "frac_hi": round(hi_counts.get("hi", 0) / n_hi, 4),
        "frac_unk_abstain": round(
            sum(hi_counts.get(t, 0) for t in ABSTAIN_TOKENS) / n_hi, 4
        ),
        "frac_other_latin": round(
            sum(hi_counts.get(k, 0) for k in LATIN_OTHER if k != "en") / n_hi, 4
        ),
        "full_distribution": {k: int(v) for k, v in hi_counts.most_common()},
    }

    # --- latency benchmark on 1000 sampled papluca_test rows ---
    rng = np.random.default_rng(12345)
    sample_idx = rng.choice(len(pap), size=1000, replace=False)
    sample_texts = [str(pap["text"].iloc[i]) for i in sample_idx]
    # warmup
    for t in sample_texts[:20]:
        pred_mod.predict(t)
    times_ms = []
    for t in sample_texts:
        t0 = time.perf_counter()
        pred_mod.predict(t)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    result["latency_ms"] = {
        "n": len(times_ms),
        "p50": round(times_ms[len(times_ms) // 2], 4),
        "p95": round(times_ms[int(0.95 * len(times_ms))], 4),
        "p99": round(times_ms[int(0.99 * len(times_ms))], 4),
        "mean": round(sum(times_ms) / len(times_ms), 4),
    }
    result["latency_pass_100ms_p99"] = result["latency_ms"]["p99"] <= 100.0

    # --- headline-vs-true gap (from result.json) ---
    rj_path = cond_dir / "result.json"
    if rj_path.exists():
        rj = json.loads(rj_path.read_text())
        result["self_reported_headline_metric"] = rj.get("headline_metric")
        result["self_reported_headline_value"] = rj.get("headline_value")
        result["headline_vs_measured_gap"] = (
            None
            if not isinstance(rj.get("headline_value"), (int, float))
            else round(abs(rj["headline_value"] - (result["papluca_test_accuracy"] or 0.0)), 6)
        )

    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
