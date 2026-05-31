# Skill: Stringify integer columns that are codes, not magnitudes

## Precondition
Apply this skill PER COLUMN. For each column `c` in train.csv, the skill applies if ALL of the following hold:

1. `train[c].dtype` is integer (`int8`, `int16`, `int32`, `int64`), AND
2. `c` is not the target, AND
3. `1 < train[c].nunique() <= max(100, 0.05 * len(train))`
   (i.e., fewer than 100 unique values OR fewer than 5% of rows are unique — too few to be a continuous magnitude), AND
4. `c` is not a row index (`c.lower() not in {"id", "index", "rowid"}` and `train[c]` is not monotonically increasing in row order), AND
5. The downstream pipeline includes ANY of: `StandardScaler`, `MinMaxScaler`, distance/kernel/linear classifier (`LogisticRegression`, `LinearRegression`, `SVC`, `KNeighborsClassifier`, `RidgeClassifier`).

If the pipeline is purely tree-based (`RandomForest`, `GradientBoosting`, `HistGradientBoosting`, `XGB`, `LightGBM`) AND uses no scaling, the skill is OPTIONAL — trees survive int-coded categoricals (with partial signal loss). Still recommended for compound codes (`re.match(r"^\d{3,}$", str(top_value))`).

## Action
- Cast the column to string in BOTH train and test BEFORE the ColumnTransformer:
  `train[c] = train[c].astype(str); test[c] = test[c].astype(str)`
- One-hot encode as a categorical going forward. Use `handle_unknown="ignore"` so test-only levels don't crash.
- If the integer has compound digit structure (e.g. `2208` = body-region 22 × severity 08), additionally split the digits into separate features.
- Record the decision in `result.json` under `audit.stringified_columns: [...]`.

## Why (failure pattern evidence)
On v1/playground-series-s3e22, A_vanilla used a `ColumnTransformer` whose numeric branch `StandardScaler()`-then-passed-through every integer column (line 47-49 of `conditions/A_vanilla/solution.py`). The dataset has a column `lesion_1` storing 4-digit anatomical codes (`2124`, `3205`, etc., ~56 unique values, cardinality 0.057 of n_rows). A standardized `lesion_1` as a number — meaning the model is now learning that "lesion code 3205 is 1.5× lesion code 2124", nonsense. B_audit stringified `lesion_1`, one-hot-encoded it, and added missing-indicator features; B's true test micro-F1 was 0.7449 vs A's 0.7206 (+0.024). The audit eval explicitly attributes the gap to `lesion_1` handling. Precondition 3 (`nunique <= 100`) catches `lesion_1`; precondition 5 catches the scaler.

## Code snippet
```python
import pandas as pd

def auto_stringify_codes(
    train: pd.DataFrame, test: pd.DataFrame, target: str,
    max_unique: int = 100, max_frac: float = 0.05,
) -> list[str]:
    """Mutates train/test in place. Returns list of columns coerced to str."""
    coerced = []
    n = len(train)
    for c in train.select_dtypes(include="integer").columns:
        if c == target or c.lower() in {"id", "index", "rowid"}:
            continue
        nuniq = train[c].nunique()
        if 1 < nuniq <= max(max_unique, int(max_frac * n)):
            # extra sanity: not monotonically increasing (== row id)
            if train[c].is_monotonic_increasing and nuniq == n:
                continue
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)
            coerced.append(c)
    return coerced

coerced = auto_stringify_codes(train, test, target="outcome")
print(f"stringified code columns: {coerced}")
```

## Cross-task evidence
- Past failure: v1/playground-series-s3e22 A_vanilla — `lesion_1` standardized as numeric; +2.4pp micro-F1 gap to B which stringified it.
- Past failure: v1/playground-series-s3e22 C_warned — same failure mode (passed `lesion_1` through HGB as int64); final true micro-F1 0.7126 (worst of three). Even with a tree model, the explicit stringification helps.
- Past failure (partial): v1/spaceship-titanic — `CabinNum` (numeric part of cabin) and `Group` (numeric prefix of PassengerId) were extracted as int by A; cardinality is higher so the precondition is less likely to fire on these, but compound-code structure (Cabin = `deck/num/side`) is exactly the case the digit-splitting clause covers.
