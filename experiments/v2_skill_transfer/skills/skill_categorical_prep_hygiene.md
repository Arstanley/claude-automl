# Skill: Categorical-prep hygiene — stringify codes, collapse rare levels, drop constants

## When this applies
Feature-engineering phase on any tabular dataset. The risks are: silently using `int`-typed code columns as numeric magnitudes, leaving constant/ID columns in the feature matrix, and one-hot-encoding categorical levels with near-singleton support.

## What to do

**1. Stringify integer codes.**
For every `int`-dtype column, check `train[col].nunique()` and `value_counts().head()`. If the integers are codes (low cardinality vs row count, composite-digit structure, ICD/SKU/anatomical/severity codes, postal codes), cast to string before encoding:
```python
train[col] = train[col].astype(str); test[col] = test[col].astype(str)
```
Then one-hot, target-encode (out-of-fold), or use `category` dtype with a tree model. For compound codes (e.g. 4-digit lesion `2208` = body-region 22, severity 08), consider digit-splitting.

**2. Decompose compound strings.**
Columns like `deck/num/side` or `firstname middlename lastname` shouldn't be fed as raw strings (5,441 unique values silently learn an ID). Split into components first.

**3. Drop constants and IDs from features.**
- `nunique() <= 1` → drop. Synthetic generators frequently preserve dead columns (`EmployeeCount=1`, `Over18='Y'`).
- Numeric monotonic `id` columns → drop from `X`. Using them as features risks ranking-by-row-order if rows were generated in label-correlated order.
- High-cardinality string identifiers (raw `Name`, `Ticket`) — drop or hash-encode; don't feed raw.

**4. Collapse rare categorical levels.**
A level with `< 5` occurrences in train (or absent from test entirely) creates CV fold instability and near-constant one-hot features. Map rare levels to `__rare__` consistently across train and test.

**5. Order ordinal-as-string columns explicitly.**
Columns like `pain`={mild, moderate, severe} have natural ordering. Define it manually; do not let `LabelEncoder` alphabetize.

## Why
Tree models partially survive `int`-coded categoricals (binary splits on values still work) but forfeit most of the categorical structure. Linear/distance/NN models treat code `3205` as 1.5× code `2200`, which is nonsense. Constants waste model capacity. Near-singleton categorical levels create unstable target encodings and rare-fold leakage. This cluster was the single biggest preprocessing win on s3e22 (+2pp micro-F1) and required for spaceship-titanic's deck-T handling.

## Code snippet
```python
# 1. Identify likely categorical-as-int columns
for col in train.select_dtypes(include="integer").columns:
    if col in {"id", TARGET}: continue
    n = train[col].nunique()
    if n < 100:
        print(f"{col}: int dtype, nunique={n}, top={train[col].value_counts().head(3).index.tolist()}")
        # If codes: train[col] = train[col].astype(str); test[col] = test[col].astype(str)

# 2. Decompose compound strings
parts = train["Cabin"].str.split("/", expand=True)
train["deck"], train["cabin_num"], train["side"] = parts[0], parts[1].astype(float), parts[2]

# 3. Drop constants + IDs
constants = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
id_cols   = [c for c in ["id","PassengerId","Id","ID"] if c in train.columns]
drop_cols = constants + id_cols
X_train = train.drop(columns=drop_cols + [TARGET])
X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])

# 4. Collapse rare categorical levels
def collapse_rare(s_tr, s_te, min_count=5):
    rare = s_tr.value_counts()[lambda c: c < min_count].index
    return (s_tr.where(~s_tr.isin(rare), "__rare__"),
            s_te.where(~s_te.isin(rare), "__rare__"))
for c in X_train.select_dtypes(include="object").columns:
    X_train[c], X_test[c] = collapse_rare(X_train[c], X_test[c])

# 5. Ordinal ordering (manual)
PAIN_ORDER = {"mild": 0, "moderate": 1, "severe": 2, "extreme": 3}
X_train["pain_ord"] = X_train["pain"].map(PAIN_ORDER)
```

## Cross-task evidence
- Saw on: playground-series-s3e22 (`lesion_1` 4-digit codes — only B stringified, gained ~2pp; `lesion_2`/`lesion_3` >99% zero near-constants), spaceship-titanic (`Cabin` is `deck/num/side` compound string with 5,441 unique levels — must decompose; deck `T` only 5 rows — only B collapsed to `__rare__`), titanic (`Ticket`/`Name` high-cardinality; `Name` contains extractable title), playground-series-s3e3 (`EmployeeCount`/`Over18`/`StandardHours` constants; many ordinal "rating" int columns), bike-sharing-demand (`weather=4` only 1 train row — must collapse or drop).
