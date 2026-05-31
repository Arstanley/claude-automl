# Skill: Treat every file the prompt names in `data/` as a training-data candidate

## Precondition
Apply this skill if ALL of the following hold:

1. The prompt explicitly names a file (or files) in the data directory beyond the
   `train.csv` / `test.csv` / `sample_submission.csv` triple, AND
2. That file is referenced with a phrase like "available for your use",
   "you may use", "additional", "supplementary", or is named but its purpose is
   NOT given (e.g., a `*_dev.txt` next to a `*_test.txt` with no schema
   description), AND
3. That auxiliary file is consistent in format with a known target class:
   - For classification: filename / content suggests it belongs to a specific
     class that is under-represented in the labeled training set
     (e.g. `hi_latn_dev.txt` aligns with a label `hi` whose train rows are
     all Devanagari; or a file `images_class_X/*.jpg` for class X).
   - For regression: the file shares the same numeric/text schema as train and
     can be concatenated.

If the file is explicitly marked "held out — do not train on", precondition is NOT met for that file. Treat any sibling file with no such warning as fair game.

## Action
- Concatenate the auxiliary file into train with the inferred label (or use
  it for self-supervised pretraining of the encoder).
- Document in `result.json` under `interpretation_notes` the exact reasoning:
  filename → assumed class → row count added.
- Re-fit and re-evaluate; the auxiliary data should improve performance on
  inputs of the matching distribution.
- If unsure whether the file is "train or evaluation", default to TRAIN on the
  dev-suffixed file and treat the test-suffixed file as evaluation-only. The
  `*_dev` / `*_test` naming convention from NLP datasets is the strongest
  signal of intent.

## Why (failure pattern evidence)
On v3/langid, the Ambig prompt said: *"`hi_latn_dev.txt` — 5,000 lines, one sentence per line. Real natural-language sentences. Available for your use."* and *"`hi_latn_test.txt` ... Held out — do not train on this."*

A_vanilla read the `*_dev` file ONLY for a probe — it computed predictions on it, observed the model was confused, and built an *abstention threshold* instead of *training on the file as `hi` data*. Result: 91.98% abstain rate on real romanized Hindi (which Adobe search would route to a degraded experience). B_audit and C_warned both trained on `hi_latn_dev.txt` as `hi` examples, achieving 98.66% and 98.78% correct routing on the held-out `hi_latn_test.txt`. The gap is enormous — A is silently useless for Hi-Latn search queries, B/C ship. The file was *named in the prompt*, *not labeled "held out"*, and *the corresponding label exists in the train set* (just in a different script). All preconditions explicitly checkable.

## Code snippet
```python
# Suppose the prompt mentioned data/hi_latn_dev.txt as "available for your use".
# Auto-detect by walking the data dir and matching against label vocabulary.
from pathlib import Path
import re
import pandas as pd

DATA = Path("data")
train = pd.read_csv(DATA / "papluca_train.csv")
labels = set(train["labels"].unique())

aux_rows = []
# Heuristic: files named "<label>_<suffix>.txt" or "<label>-<suffix>.txt"
# where <label> ∈ trained labels are candidate auxiliary training data.
for path in DATA.glob("*.txt"):
    stem = path.stem
    # match prefix label, e.g. "hi_latn_dev" -> "hi", "fr_news" -> "fr"
    m = re.match(r"^([a-z]{2,3})_", stem)
    if not m or m.group(1) not in labels:
        continue
    if "test" in stem.lower():
        continue  # respect held-out markers
    label = m.group(1)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    aux_rows.extend({"labels": label, "text": l} for l in lines)
    print(f"  + {path.name}: {len(lines)} rows as label={label}")

if aux_rows:
    aux = pd.DataFrame(aux_rows)
    train = pd.concat([train, aux], ignore_index=True)
    print(f"train grew from {len(train) - len(aux)} -> {len(train)} rows")
```

## Cross-task evidence
- Past failure: v3/langid A_vanilla — never trained on `hi_latn_dev.txt`; 91.98% abstain rate on the held-out romanized Hindi probe vs 98.7%+ correct for B/C which did train on it. The single largest behavioral gap across all our experiments.
