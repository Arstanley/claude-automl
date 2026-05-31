# Skill: Report trivial baselines and exploit known invariants

## When this applies
Every modeling task. The baseline floor tells you whether your model has earned its complexity. Invariants (deterministic relationships between columns) are free signal that beats any model on a subset of rows.

## What to do
- Compute and report at least these baselines before reporting your model's score:
  - **Majority-class baseline** (classification): predict the most common class. Sets the floor.
  - **Marginal baseline** (regression): predict the mean / median of the train target.
  - **Single-feature baselines**: predict from one obviously-strong feature (e.g. gender in titanic, hour-of-day in bike-sharing).
  - **Sample-submission baseline** (if it has values): some sample submissions encode a non-trivial baseline (e.g. titanic's `gender_submission` is the gender heuristic ~78% accuracy).
- If your tuned model doesn't beat the strongest baseline, the complexity is unjustified — investigate before submitting.
- Look for **deterministic invariants** in the data documentation or by inspection:
  - "X = Y + Z" identities (e.g. `count == casual + registered`).
  - "If A then B" rules (e.g. `CryoSleep=True ⇒ all amenity spend = 0`).
  - Constant-implied invariants (a flag column that's always 1 when another column is null).
- Use invariants for:
  - **Imputation**: fill missing values deterministically when the invariant applies.
  - **Data validation**: flag rows that violate the invariant as data-quality issues.
  - **Decomposition**: if `count = casual + registered`, modeling the two heads separately and summing can outperform direct modeling of `count` (or not — test both).
- Don't "discover" invariants stated in the spec and claim them as feature engineering wins; just use them.

## Why
Without baselines, "0.82 accuracy" is meaningless — you don't know if the model added 4pp or 40pp over trivial. Invariants are deterministic signal that no model can improve on; ignoring them leaves accuracy on the floor (saw on spaceship-titanic: only B_audit used CryoSleep⇒zero-spend bidirectionally for imputation).

## Code snippet
```python
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# Classification baselines
maj = y_train.mode().iloc[0]
print(f"majority-class acc: {(y_val == maj).mean():.4f}")
# Single-feature baseline (e.g. gender on titanic)
gender_pred = (X_val["Sex"] == "female").astype(int)
print(f"gender-heuristic acc: {accuracy_score(y_val, gender_pred):.4f}")

# Regression baseline
print(f"marginal-mean RMSE: {((y_val - y_train.mean())**2).mean()**0.5:.4f}")

# Identity check (bike-sharing example)
assert (train["count"] == train["casual"] + train["registered"]).all(), "Identity broken"

# Invariant-based imputation (spaceship-titanic example)
zero_spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
all_zero = (train[zero_spend_cols].sum(axis=1) == 0)
# CryoSleep=True forces zero spend; reverse direction: zero spend ⇒ very likely CryoSleep
train.loc[train["CryoSleep"].isna() & all_zero, "CryoSleep"] = True
```

## Cross-task evidence
- Saw on: titanic (gender baseline ~78%; B_audit reported it, A_vanilla did not), spaceship-titanic (CryoSleep ⇒ zero-spend; only B used it bidirectionally), bike-sharing-demand (`count = casual + registered` identity; two-head decomposition won on the actual test set when validated correctly), playground-series-s3e3 (class imbalance ~11.6% → constant 0.1193 baseline gives AUC=0.5 floor), playground-series-s3e22 (majority class `lived` = 45.8% accuracy floor).
