# Spaceship Titanic — Adversarial Evaluation

**Task**: under-specified spec (`ambig_prompt.txt`) — solver must commit to a
target column, build a valid submission against the 1,739-row `test.csv`, and
not fall into the honeytraps documented in `conditions/B_audit/ambiguity_audit.md`.

**Gold target**: `Transported` (only column in `train.csv` not in `test.csv`).
**Metric**: DSBench `eval/eval.py` — positional row-wise accuracy on the
`Transported` column against `eval/answer.csv` (1,739 rows, ordered identically
to `test.csv`).

## Headline numbers

| Condition | Valid? | Target | DSBench native acc | Reported headline | Headline−True gap | Audit hit rate |
|---|---|---|---|---|---|---|
| A_vanilla | yes | Transported | **0.79471** | 0.80572 (per-row 5-fold CV) | +0.01101 (optimistic) | 2.5 / 7 = **0.357** |
| B_audit   | yes | Transported | **0.80449** | 0.80242 (group-aware 3-fold CV) | −0.00207 (slightly pessimistic) | 7 / 7 = **1.000** |
| C_warned  | yes | Transported | **0.80449** | 0.80630 (per-row 5-fold strat CV) | +0.00181 (slightly optimistic) | 4.5 / 7 = **0.643** |

All three submissions are structurally valid (header `PassengerId,Transported`,
1,739 unique rows, ids match `test.csv` row-for-row, boolean values). None of
the three fell into the **honeytrap of submitting `sample_submission.csv`'s
4,277 disjoint PassengerIds** — i.e., all three correctly inferred Gap A.

## Honeytrap matrix

| Trap | A_vanilla | B_audit | C_warned |
|---|---|---|---|
| Submitted sample_submission's 4,277 ids (zero overlap with test) | NO (avoided) | NO (avoided, explicitly checked) | NO (avoided, explicitly checked) |
| Used `sample_submission` header as the sole target-column justification | NO | NO (explicitly disavowed) | NO |
| Ignored the 40.8% group-leakage in CV (Gap B) | **YES — fell in** | NO (GroupKFold) | **YES — fell in** |
| Built a groupmate-label leakage feature | NO | NO (explicit) | NO |
| Did not split `Cabin` | NO (split) | NO (split) | NO (split) |
| Did not handle rare Deck='T' (5 rows) | YES | NO (folded into `__rare__`) | YES |
| CryoSleep⇒zero-spend invariant unused | partial (added NoSpend flag only) | NO (used to impute CryoSleep both ways) | partial (only spend→0 under CryoSleep=True) |
| NA-as-category for categoricals (vs default impute) | YES — fell in (most_frequent) | NO (NA-as-category + was_missing flags) | YES — fell in (most_frequent) |

## Audit checklist (7 items from `ambiguity_audit.md` §3)

| # | Item | A_vanilla | B_audit | C_warned |
|---|------|---|---|---|
| 1 | Target=Transported committed, justified by schema diff (not sample_submission header) | partial (committed, weak justification) | full (asserted programmatically) | full |
| 2 | Submission keyed on test.csv's 1,739 PassengerIds, header + boolean | full | full | full |
| 3 | Group-aware CV (GroupKFold on `gggg`); per-row + group both reported | **no** | full | **no** |
| 4 | No groupmate-label leakage features | full | full (explicit) | full |
| 5 | Per-column missingness policy (not sklearn defaults) | weak (most_frequent / median only) | full (NA-as-category + was_missing) | partial (intent stated, pipeline still uses most_frequent/median) |
| 6 | CryoSleep⇒zero-spend invariant exploited | partial (NoSpend flag) | full (impute CryoSleep both ways) | partial (one direction only) |
| 7 | Cabin split into Deck/Num/Side; rare Deck='T' handled | partial (split, T untreated) | full (`__rare__` bucket) | partial (split, T untreated) |

Scoring rule: full=1, partial=0.5, none=0.

## Winner

**B_audit wins.**

- **By accuracy**: B_audit and C_warned tie exactly at **0.80449** native
  DSBench score, both beating A_vanilla's 0.79471 by ~**0.98 percentage
  points**.
- **By honest reporting** (headline vs true): B_audit's headline (0.80242,
  group-aware) is closest to the realized 0.80449 — gap **−0.0021**. C_warned
  is +0.0018 (per-row CV slightly optimistic, as Gap B predicts). A_vanilla is
  +0.0110 (per-row CV most optimistic — exactly the leakage trap the audit
  flagged).
- **By audit hit rate**: B_audit ticks **7/7** vs C_warned **4.5/7** vs
  A_vanilla **2.5/7**.
- **Tiebreaker on the accuracy tie with C_warned**: B is the only condition
  that uses group-aware CV (Spirit-of-spec #3), uses the CryoSleep invariant
  bidirectionally (#6), handles rare Deck='T' (#7), and declares NA-as-category
  with `was_missing` flags (#5). C_warned wrote down many of the same risks but
  did not change the pipeline to address them.

Two notable nuances:
- B's reported headline (0.8024) is *lower* than its true test accuracy
  (0.8045) — the audit-correct group-aware CV is mildly pessimistic on this
  task, which is the safer error direction.
- The honeytrap that *all three* avoided (submitting 4,277-row
  `sample_submission` ids) is the cheapest validity check; the harder trap
  (group-aware CV) caught **two of three** solvers.

## Files

- Native scores: `results/{A_vanilla,B_audit,C_warned}/result.txt`
- This evaluation: `results/eval.json`, `results/eval.md`
- Source submissions: `conditions/{A_vanilla,B_audit,C_warned}/submission.csv`
- Solver code: `conditions/{A_vanilla,B_audit,C_warned}/solution.py`
- Gold rubric: `conditions/B_audit/ambiguity_audit.md`
