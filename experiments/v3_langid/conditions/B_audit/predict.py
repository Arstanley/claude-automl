#!/usr/bin/env python3
"""
predict.py — loads model.pkl and exposes `predict(text: str) -> str`.

Used by the evaluator to benchmark latency. Hot path is single-threaded,
batch-size 1, and warm.

Also exposes:
    predict_with_conf(text)         -> (label, confidence)
    predict_strict(text, tau=0.40)  -> label or "unk"  (abstention API)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

# Must import the class so unpickling resolves it. solution.py defines it,
# and solution.py rebinds LangIDModel.__module__ = "solution" before saving.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solution  # noqa: F401  -- registers LangIDModel + _script_route under module "solution"

_MODEL_PATH = HERE / "model.pkl"
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL


def predict(text: str) -> str:
    """Return the predicted language code for `text` (one of the 20 labels)."""
    return _get_model().predict(text)


def predict_with_conf(text: str):
    return _get_model().predict_with_conf(text)


def predict_strict(text: str, tau: float = 0.40) -> str:
    return _get_model().predict_strict(text, tau=tau)


if __name__ == "__main__":
    # tiny smoke test
    for t in [
        "Hello world",
        "Bonjour le monde",
        "iske aane se purva hi log gharon ki safai ka karya shuru kar dete hain.",
        "размерът на хоризонталната мрежа",
        "你好世界",
        "こんにちは世界",
        "مرحبا بالعالم",
    ]:
        print(f"{t!r:80s} -> {predict(t)}   ({predict_with_conf(t)})")
