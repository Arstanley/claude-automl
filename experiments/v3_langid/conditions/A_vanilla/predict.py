#!/usr/bin/env python3
"""Load model.pkl and expose predict(text: str) -> str.

Returns 'unk' if the top-class probability is below the bundled confidence
threshold (the "must not be confidently wrong" hard rule from the spec).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / "model.pkl"

_bundle = joblib.load(_MODEL_PATH)
_pipe = _bundle["pipeline"]
_classes = np.array(_bundle["classes"])
_threshold = float(_bundle["conf_threshold"])


def predict(text: str) -> str:
    proba = _pipe.predict_proba([text])[0]
    top = int(proba.argmax())
    if float(proba[top]) < _threshold:
        return "unk"
    return str(_classes[top])


if __name__ == "__main__":
    samples = [
        "Hello, how are you today?",
        "Bonjour tout le monde",
        "你好，世界",
        "iske aane se purva hi log gharon ki safai ka karya shuru kar dete hain.",
        "",
    ]
    for s in samples:
        print(repr(s[:60]), "->", predict(s))
