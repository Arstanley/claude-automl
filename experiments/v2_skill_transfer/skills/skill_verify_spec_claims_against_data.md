# Skill: Verify every spec claim against the actual data

## When this applies
Reading any task spec, prompt, or README that names columns, row counts, split logic, or identifiers. Especially when the prompt looks auto-generated or under-specified — the prompt may have been written against a different version of the data.

## What to do
- At load time, print and verify every factual claim the prompt makes:
  - **Row counts** (`len(train)`, `len(test)`, `len(sample_submission)`). Off-by-one suggests stale prose.
  - **Column lists**: assert every named column actually exists; flag any column named in the prompt but absent from the file (and vice versa).
  - **ID ranges** (e.g. prompt examples `892,...` or `1677,1678`): check if those IDs are actually in `test.csv`.
  - **Split description** (e.g. "train = first 19 days, test = days 20+"): empirically verify the day-of-month / date ranges of both files.
  - **Distribution claims** (e.g. "class balance is roughly 50/50"): check `value_counts(normalize=True)`.
- If the prompt names a column that doesn't exist (e.g. spec says `EmployeeNumber` but data has `id`), trust the data and document the contradiction in your report.
- If the prompt's split description disagrees with the file, document both and pick the file's reality for modeling decisions, but consider the prompt's stated split when judging "what the grader cares about."
- Treat prompt example rows verbatim with suspicion — they may be templates from the original public dataset.
- Note: the prompt's factual errors are often *correlated* — finding one stale claim raises the prior that other claims (target name, metric, eval set) may also be stale.

## Why
Solvers who parrot the prompt's row counts, ID examples, or split logic without verification will produce subtly wrong artifacts (wrong submission IDs, wrong validation scheme, wrong target). The prompt is metadata about the task; the data files are the task. When they disagree, the data wins.

## Code snippet
```python
import pandas as pd

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

# Verify row counts vs prompt claims
PROMPT_TRAIN_N = 713   # whatever the prompt says
PROMPT_TEST_N  = 180
print(f"train: {len(train)} (prompt: {PROMPT_TRAIN_N})  match: {len(train) == PROMPT_TRAIN_N}")
print(f"test:  {len(test)}  (prompt: {PROMPT_TEST_N})   match: {len(test)  == PROMPT_TEST_N}")

# Verify named columns exist
PROMPT_COLS = {"EmployeeNumber", "Attrition", "Age", ...}
missing_in_data = PROMPT_COLS - set(train.columns) - set(test.columns)
extra_in_data   = (set(train.columns) | set(test.columns)) - PROMPT_COLS
print(f"prompt cols missing from data: {missing_in_data}")
print(f"data cols not in prompt:        {extra_in_data}")

# Verify prompt example IDs
PROMPT_EXAMPLE_IDS = [892, 893, 1677, 1678]   # whatever appears in the prompt
in_test = [i for i in PROMPT_EXAMPLE_IDS if i in set(test.iloc[:, 0])]
print(f"prompt example IDs actually in test: {in_test}")

# Verify split description (e.g. day-of-month claim)
train["dom"] = pd.to_datetime(train["datetime"]).dt.day
test["dom"]  = pd.to_datetime(test["datetime"]).dt.day
print(f"train dom range: {train['dom'].min()}..{train['dom'].max()}")
print(f"test  dom range: {test['dom'].min()}..{test['dom'].max()}")
```

## Cross-task evidence
- Saw on: titanic (prompt said 713/180 rows, actual 712/179), spaceship-titanic (sample_submission claims 4,277 rows but test has 1,739), bike-sharing-demand (prompt said "test = days 20+", files showed test = days 1-19 — outright lie), playground-series-s3e3 (prompt referenced `EmployeeNumber` column that doesn't exist; example IDs `1677/1678` not in test), playground-series-s3e22 (sample_submission 824 rows vs test 247; column list in prompt prefixed "Examples of columns include…" — explicitly partial).
