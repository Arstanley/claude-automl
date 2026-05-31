# Skill: Probe ID columns for parseable group structure before choosing CV

## Precondition
Apply this skill if ANY of the following is true after loading train.csv and test.csv:

1. There exists a column `c` (other than the target) such that
   `len(set(train[c].astype(str)) & set(test[c].astype(str))) / max(1, test[c].nunique()) >= 0.05`
   (i.e., at least 5% of unique test values for that column also appear in train), OR
2. There exists a string-dtype column whose values are non-empty AND `>= 80%`
   of them match the regex `^[A-Za-z0-9]+[_\-/][A-Za-z0-9]+$` (a parseable
   compound ID, e.g. `0001_01`, `cabin/12/P`), OR
3. There exists an int-dtype "id"-like column (name contains `id`, `number`,
   `member`, `patient`) whose value range overlaps train and test by `>= 5%`.

If none of (1)/(2)/(3) holds, this skill does NOT apply — random KFold/StratifiedKFold is fine.

## Action
- Compute the train/test overlap fraction on the suspected group column. Print it. Persist it in `result.json` under `audit.group_leakage_probe`.
- If the column has compound structure (precondition 2), split on the delimiter and treat the first component as the group: e.g. for `PassengerId = "gggg_pp"`, `groups = train["PassengerId"].str.split("_").str[0]`.
- Switch CV to `GroupKFold` (or `StratifiedGroupKFold` for classification) keyed on the group column.
- Report BOTH per-row stratified CV and group-aware CV — the gap = your leakage size. Make group-aware the headline.

## Why (failure pattern evidence)
On v1/spaceship-titanic, A_vanilla parsed `PassengerId` into `Group/GroupSize` AS FEATURES but ran a plain `cross_val_score(pipe, X, y, cv=5)` (line 90 of `conditions/A_vanilla/solution.py`). The dataset has 40.8% of test rows sharing a `gggg` group with train. A's reported CV accuracy was 0.8057 (5-fold stratified); the true test accuracy was 0.7947 — a +0.011 optimistic gap that exactly matches the leakage rate. B_audit applied GroupKFold and reported 0.8024 (CV) → 0.8045 (true), gap −0.002. The fix is mechanical and free; the failure mode is that A built the group feature but never probed whether the group leaked into CV.

## Code snippet
```python
import re
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

GROUP_LEAK_THRESHOLD = 0.05  # 5% of test rows share an id with train -> material

def find_group_columns(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    """Return {column_name: leakage_fraction} for columns flagged by precondition."""
    flags = {}
    compound_re = re.compile(r"^[A-Za-z0-9]+[_\-/][A-Za-z0-9]+")
    for c in train.columns:
        if c == target or c not in test.columns:
            continue
        a, b = set(train[c].astype(str)), set(test[c].astype(str))
        if not b:
            continue
        leakage = len(a & b) / len(b)
        is_compound = (
            train[c].dtype == object
            and train[c].astype(str).str.match(compound_re).mean() > 0.8
        )
        if leakage >= GROUP_LEAK_THRESHOLD or is_compound:
            flags[c] = {"leakage_frac": leakage, "compound": bool(is_compound)}
    return flags

flags = find_group_columns(train, test, target="Transported")
print("group-leakage probe:", flags)

# If any flagged, build groups and use GroupKFold
if flags:
    group_col = max(flags, key=lambda c: flags[c]["leakage_frac"])
    if flags[group_col]["compound"]:
        groups = train[group_col].astype(str).str.split("_").str[0]
    else:
        groups = train[group_col]
    cv = GroupKFold(n_splits=5)
    cv_iter = cv.split(X, y, groups)
else:
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_iter = cv.split(X, y)
```

## Cross-task evidence
- Past failure: v1/spaceship-titanic A_vanilla — +0.011 optimistic CV vs true test (0.8057 → 0.7947); only B used GroupKFold and reported a calibrated 0.8024.
- Past failure: v1/playground-series-s3e22 A_vanilla — `hospital_number` has 90% train/test overlap; A correctly dropped it as a feature but did not switch CV; A's headline 0.7055 vs true 0.7206 still under-promised, but the failure mode is the same shape (probing only "use as feature" without probing "leak into CV").
