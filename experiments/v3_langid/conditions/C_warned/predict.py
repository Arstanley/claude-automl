#!/usr/bin/env python3
"""
Load model.pkl and expose predict(text: str) -> str.

Implements the 'must not be confidently wrong' guard via two abstain rules:
  - <3 alpha characters in the input -> return 'unk'
  - top-1 prob < abstain_prob AND (top1-top2) margin < abstain_margin
    -> return 'unk'

The label space is the 20 papluca languages
(ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh)
plus the sentinel 'unk' for abstained inputs.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _BLOB = pickle.load(_f)

_PIPE = _BLOB["pipeline"]
_CLASSES = np.asarray(_BLOB["classes"])
_ABSTAIN_PROB = float(_BLOB.get("abstain_prob", 0.40))
_ABSTAIN_MARGIN = float(_BLOB.get("abstain_margin", 0.10))
_MIN_ALPHA = int(_BLOB.get("min_alpha_chars", 3))

_ALPHA_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)


def predict(text: str) -> str:
    """Return ISO-639-1 language code, or 'unk' if the model would be
    confidently wrong / has no signal."""
    if text is None:
        return "unk"
    s = str(text)
    if len(_ALPHA_RE.findall(s)) < _MIN_ALPHA:
        return "unk"
    proba = _PIPE.predict_proba([s])[0]
    order = np.argsort(proba)[::-1]
    top1_p = float(proba[order[0]])
    top2_p = float(proba[order[1]])
    if top1_p < _ABSTAIN_PROB and (top1_p - top2_p) < _ABSTAIN_MARGIN:
        return "unk"
    return str(_CLASSES[order[0]])


if __name__ == "__main__":
    # quick smoke test
    for ex in [
        "Hello, how are you?",
        "Bonjour, comment ca va?",
        "Hola, como estas?",
        "नमस्ते दुनिया",
        "Namaste duniya kaise ho aap",
        "你好世界",
        "",
        "ok",
        "12345",
    ]:
        print(repr(ex), "->", predict(ex))
