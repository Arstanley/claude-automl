# Skill: Model missingness as signal, not noise

## When this applies
Any column with non-trivial NaN rate (≥1% is worth thinking about; ≥5% is mandatory to handle deliberately). Especially when the "field not recorded" condition is plausibly correlated with the label (medical, financial, behavioral records).

## What to do
- Run a missingness profile at load time: `train.isna().mean().sort_values(ascending=False)` and the same on test. Compare — diverging missingness between train and test is a distribution-shift flag.
- For each column with non-trivial NaN rate, choose a policy explicitly. Do NOT rely on `sklearn` defaults (which silently drop, mean-impute, or fail).
- **Add missing-indicator features** alongside imputation for high-missingness columns:
  ```python
  X["age_was_missing"] = X["age"].isna().astype(int)
  X["age"] = X["age"].fillna(X["age"].median())
  ```
  Trees can use the indicator as a split; the imputed value carries the central tendency.
- For categorical columns, treat `NaN` as its own category (`"__missing__"`) rather than imputing the mode — the absence itself is information.
- Watch for **invariant-implied missingness**: if a row's other columns logically determine the missing value (e.g. spaceship-titanic's `CryoSleep=True` ⇒ all amenity spend = 0), use the invariant to impute deterministically.
- Watch for **encoded missingness**: values like `0` or `-1` in a sensor column that actually mean "not measured" (saw `windspeed=0` in bike-sharing as a sensor-floor artifact, `humidity=0` similarly).

## Why
Silent default imputation throws away the signal in *whether* the field was recorded, which often correlates with the label more than the imputed value does. Treating sensor-floor zeros as legitimate measurements distorts model fit. Mode/median imputation also assumes MCAR, which is rarely true on real records.

## Code snippet
```python
miss = train.isna().mean().sort_values(ascending=False)
print(miss[miss > 0].to_string())

# Per-column policy
HIGH_MISSING = miss[miss > 0.05].index.tolist()
for col in HIGH_MISSING:
    X[f"{col}_was_missing"] = X[col].isna().astype(int)
    if X[col].dtype == "object":
        X[col] = X[col].fillna("__missing__")
    else:
        X[col] = X[col].fillna(X[col].median())

# Invariant-implied imputation (example)
zero_spend = (train[["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]].sum(axis=1) == 0)
train.loc[train["CryoSleep"].isna() & zero_spend, "CryoSleep"] = True

# Sensor-floor flag
train["windspeed_is_floor"] = (train["windspeed"] == 0).astype(int)
```

## Cross-task evidence
- Saw on: playground-series-s3e22 (vet records: `abdomen` 17.6% missing, `rectal_exam_feces` 15.3% — only B_audit added indicators and gained), spaceship-titanic (CryoSleep ⇒ zero-spend invariant, ~2% missing across all columns is informative), titanic (`Age` ~20% missing with `xx.5` decimal as estimation flag — bucketing destroys signal), bike-sharing-demand (`windspeed==0` and `humidity==0` are sensor artifacts, 12% of rows).
