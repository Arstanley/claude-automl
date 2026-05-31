# Skill: Verify the prompt's claim about the train/test split before designing your holdout

## Precondition
Apply this skill if BOTH of the following hold:

1. The prompt contains a structural claim about how train and test were split — patterns to detect:
   - Phrase contains any of: "days N+", "after [date/year]", "future", "held out", "earliest N", "first N", "20% holdout", "split chronologically", "split by [column]", or any specific numeric or temporal boundary, AND
2. The claim is empirically checkable by reading train.csv and test.csv:
   - Has a column with dtype datetime / parseable date, OR has a temporal column the prompt mentions (e.g., `datetime`, `date`, `year`, `dteday`), OR
   - Has a group/ID column the claim references.

When met: write a 5-line probe that compares the actual distribution to the claimed split BEFORE picking your validation scheme.

## Action
- Parse the claim into a checkable predicate (e.g., "test.date.min() >= train.date.max()" or "test.day_of_month > 19").
- Run the predicate on the actual data. Print the result.
- If FALSE: design your validation to match the ACTUAL split, not the claimed one. Specifically:
  - If train/test date ranges overlap → use random `KFold` / `StratifiedKFold` matching the actual interleaving, NOT a strict temporal holdout.
  - If group claim is false → see `skill_probe_group_structure_in_ids`.
- Persist `{"split_claim": "...", "claim_verified": true/false, "actual_distribution": {...}}` in `result.json`.
- Lock in the empirically-correct CV; if the prompt and data conflict, the data wins.

## Why (failure pattern evidence)
On v1/bike-sharing-demand, the Ambig prompt claimed: *"Train = first 19 days of each month, test = days 20+."* A_vanilla believed this and built a days-16-19 holdout. The shipped test set is actually days 1-19 (random within-day interleaving with train). A's headline RMSLE was 0.309 on its strict holdout; true test RMSLE was 0.279 — a −0.029 conservative gap, but the deeper problem is that the strict holdout pushed A toward defensive feature choices and a single-head regression on log1p(count). C_warned's looser time-based last-20% holdout was much better calibrated to the actual test (true 0.271, headline 0.306) — and C won by 0.008 RMSLE because its less-pessimistic validation kept the two-head (`casual`+`registered`) approach in play. The prompt LIED about the split; A trusted the prompt; A lost.

The verification is one line: `train["datetime"].dt.day.max(), test["datetime"].dt.day.max()` → both 19, contradicting the prompt.

## Code snippet
```python
import re
import pandas as pd

def verify_split_claim(prompt: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Return audit dict; print warnings if claims contradict data."""
    audit = {}
    # 1. Look for a datetime column
    dt_col = None
    for c in train.columns:
        if c in test.columns:
            try:
                pd.to_datetime(train[c].iloc[:5])
                dt_col = c
                break
            except Exception:
                continue
    if dt_col:
        tr_dt = pd.to_datetime(train[dt_col])
        te_dt = pd.to_datetime(test[dt_col])
        audit["train_date_range"] = (str(tr_dt.min()), str(tr_dt.max()))
        audit["test_date_range"] = (str(te_dt.min()), str(te_dt.max()))
        audit["train_day_max"] = int(tr_dt.dt.day.max())
        audit["test_day_max"] = int(te_dt.dt.day.max())
        audit["ranges_overlap"] = bool(tr_dt.max() >= te_dt.min() and te_dt.max() >= tr_dt.min())
        # check claims like "test is days 20+"
        m = re.search(r"days?\s+(\d+)\s*\+", prompt, re.IGNORECASE)
        if m:
            claimed_min = int(m.group(1))
            audit["claim_min_day"] = claimed_min
            audit["claim_holds"] = audit["test_day_max"] >= claimed_min and audit["train_day_max"] < claimed_min
            if not audit["claim_holds"]:
                print(f"WARNING: prompt claims test starts at day {claimed_min}, "
                      f"but actual test_day_max={audit['test_day_max']} "
                      f"and train_day_max={audit['train_day_max']}. Trust data, not prompt.")
    return audit

# usage
audit = verify_split_claim(open("prompt.md").read(), train, test)
# Use audit["ranges_overlap"] etc to pick CV
```

## Cross-task evidence
- Past failure: v1/bike-sharing-demand A_vanilla — believed prompt's "days 20+" claim, built days-16-19 holdout, true RMSLE 0.279 vs winner C's 0.271. A never checked `train["datetime"].dt.day.max()`.
- Past failure: v1/bike-sharing-demand B_audit — even with audit, B's defensive strict late-days-of-month holdout caused it to discard the better two-head model (loses to C by 0.004 RMSLE). The audit was right to flag the prompt-vs-data mismatch but wrong to over-correct. The skill explicitly says "match the ACTUAL split, not the claimed one" — which on bike-sharing means a random-style holdout, NOT a strict temporal one.
