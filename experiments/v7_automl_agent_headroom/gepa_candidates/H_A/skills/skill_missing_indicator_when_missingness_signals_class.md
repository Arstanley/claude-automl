# Skill: Add explicit missing-indicator features when missingness rate per column exceeds 5%

## Precondition
Apply this skill PER COLUMN. For each column `c` in train.csv, the skill applies if ALL of the following hold:

1. `c` is not the target and not an ID column, AND
2. `train[c].isna().mean() >= 0.05` (≥5% missing in train), AND
3. EITHER the missingness rate differs between train and test by more than 2pp
   (`abs(train[c].isna().mean() - test[c].isna().mean()) > 0.02`),
   OR the column is one of the predictive features (not an audit-only field), AND
4. The pipeline currently imputes silently with `SimpleImputer(strategy="median"|"mean"|"most_frequent")` or equivalent.

Bonus trigger (apply even if rate < 5%): if `train[c]` is integer/float AND contains a "sentinel" value pattern (`-1`, `9999`, `-9999`, `0` in a column whose mean is positive — e.g. `weather=4` only 1 row) treat that value as missing as well.

## Action
- For each flagged column, add a binary indicator feature: `train[c + "_missing"] = train[c].isna().astype(int)` (mirror in test).
- Then impute the original column. The model now has BOTH the imputed value AND the "was missing" signal.
- For categorical columns, prefer `NA-as-category` (fill NaN with the literal string `"__missing__"` before one-hot) rather than `most_frequent` — the missing-class is often predictive.
- For numeric sentinels: convert `-1`/`9999`/etc. to `NaN` first, then apply the above.
- Record in `result.json` under `audit.missingness_indicators_added: [...]`.

## Why (failure pattern evidence)
On v1/playground-series-s3e22 (Horse Survival), several columns have 30-60% missingness (e.g., `rectal_temp`, `pulse`, `peripheral_pulse`, `mucous_membrane`). A_vanilla used `SimpleImputer(strategy="median")` for numeric and `most_frequent` for categorical with NO indicator features (lines 35-47 of `conditions/A_vanilla/solution.py`). B_audit added `*_was_missing` indicators per high-missingness column and treated NaN as its own category for categoricals. B's true micro-F1 was 0.7449 vs A's 0.7206 — a +0.024 gap, attributed in the eval to the combination of `lesion_1` stringification AND missing-indicators. In medical/biological data, missingness IS informative ("the vet didn't measure rectal temperature" often correlates with "horse was already in distress"); silent imputation throws this signal away.

The precondition is mechanically checkable: `train.isna().mean()` → flag any column ≥ 0.05.

## Code snippet
```python
import pandas as pd
import numpy as np

SENTINEL_VALUES_NUMERIC = {-1, -9999, 9999, 999}

def add_missingness_features(
    train: pd.DataFrame, test: pd.DataFrame, target: str,
    min_rate: float = 0.05,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    indicators_added = []
    n_tr = len(train)
    for c in train.columns:
        if c == target or c.lower() in {"id", "index"}:
            continue
        # Convert numeric sentinels to NaN
        if pd.api.types.is_numeric_dtype(train[c]):
            mask = train[c].isin(SENTINEL_VALUES_NUMERIC)
            if mask.any() and mask.mean() < 0.5:  # sentinels, not encoded baseline
                train.loc[mask, c] = np.nan
                if c in test.columns:
                    test_mask = test[c].isin(SENTINEL_VALUES_NUMERIC)
                    test.loc[test_mask, c] = np.nan
        miss_rate = train[c].isna().mean()
        if miss_rate >= min_rate:
            ind = f"{c}_missing"
            train[ind] = train[c].isna().astype(int)
            if c in test.columns:
                test[ind] = test[c].isna().astype(int)
            indicators_added.append(c)
    # Categorical NaN -> '__missing__' literal
    for c in train.select_dtypes(include="object").columns:
        if c in indicators_added:
            train[c] = train[c].fillna("__missing__")
            if c in test.columns:
                test[c] = test[c].fillna("__missing__")
    return indicators_added, train, test

added, train, test = add_missingness_features(train, test, target="outcome")
print(f"added missingness indicators for: {added}")
```

## Cross-task evidence
- Past failure: v1/playground-series-s3e22 A_vanilla — silent median/most_frequent imputation; only B added `*_missing` indicators and NA-as-category. +0.024 micro-F1 gap (attributed by audit eval to indicators + lesion_1 fix combined).
- Past failure: v1/spaceship-titanic A_vanilla — same shape: `SimpleImputer(median/most_frequent)` with no indicators. B added `was_missing` flags on `Age`, `CryoSleep`, `HomePlanet`, etc. A loses by +0.0098 on true accuracy.
- Past failure: v1/bike-sharing-demand — `weather=4` is a sentinel-like value with only 1 train occurrence. Only B explicitly collapsed/handled it. (Bonus trigger covers this case.)
