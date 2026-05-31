# Skill: Identify target by train/test schema diff

## When this applies
The task prompt is ambiguous, partial, or contradictory about which column to predict (uses placeholders like `<target>`, lists multiple label-like columns, or names a column that doesn't exist). Any time a competition-style task has separate train and test CSVs.

## What to do
- Compute `set(train.columns) - set(test.columns)` at load time. The structurally absent column(s) are the only viable target candidates.
- If exactly one column is missing from test, commit to it in writing and cite the diff as the justification.
- If multiple columns are missing (e.g. `casual`, `registered`, `count` in bike-sharing), cross-check the metric, the sample-submission header, and which column has a defined "submission unit" to disambiguate. Document the rejection of the others.
- Do NOT justify the target choice from prompt prose, cultural priors ("everyone knows this is the Titanic task"), or the sample-submission header alone. The schema diff is the load-bearing check; everything else is corroboration.
- Reject any candidate that is present in `test.csv` — predicting an already-observed column is either trivial or nonsensical.

## Why
Prompts may use placeholder names (`<target>`), reference columns that don't exist (`EmployeeNumber`), or omit the target name entirely. Schema-diff is the only signal in the data itself. Picking the wrong target produces a confident-looking but unscorable submission.

## Code snippet
```python
import pandas as pd

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

candidates = set(train.columns) - set(test.columns)
assert len(candidates) >= 1, "No column is held out from test — re-read the task."
print(f"Target candidates (train-only columns): {candidates}")

# If multiple, narrow by metric compatibility:
# - regression metric  -> numeric continuous candidate
# - AUC / log-loss     -> binary candidate
# - F1/accuracy multiclass -> low-cardinality categorical candidate
TARGET = "..."   # commit explicitly; record this in the run report
assert TARGET in candidates, f"{TARGET} is not in train-only columns {candidates}"
```

## Cross-task evidence
- Saw on: titanic (`Survived` only train-only column), spaceship-titanic (`Transported`), bike-sharing-demand (3 candidates `casual`/`registered`/`count` — disambiguated by metric+sample-sub header), playground-series-s3e3 (`Attrition`; prompt referenced bogus `EmployeeNumber`), playground-series-s3e22 (`outcome` among many distractors).
