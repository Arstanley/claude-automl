# Titanic evaluation — A_vanilla vs B_audit vs C_warned

Gold target: `Survived`. Gold ID: `PassengerId`. Eval: classification accuracy
on 179 held-out rows (PassengerId 6..890), via `eval/eval.py`.

## Results

| Condition | Target | Valid submission | DSBench acc | Audit coverage | Headline | Headline − true |
|---|---|---|---|---|---|---|
| A_vanilla | Survived | yes (header, 179 rows, test IDs in order, int 0/1) | **0.8268** | 5/8 (0.625) | 0.8216 (CV5) | -0.0052 |
| B_audit | Survived | yes (same) | **0.8268** | 8/8 (1.000) | 0.8203 (CV5) | -0.0066 |
| C_warned | Survived | yes (same) | **0.8324** | ~6/8 (0.750) | 0.8314 (CV5) | -0.0010 |

## Honeytraps

None tripped by any solver: nobody used the `gender_submission.csv`
PassengerIds 892..1309 (the audit's main trap), all wrote literal
`PassengerId,Survived` header with integer 0/1 values in test.csv order.

## Audit-checklist detail

| Item | A | B | C |
|---|---|---|---|
| 1. Target by structural diff, not folklore | partial (comment only) | yes (runtime set-diff) | yes (runtime assert) |
| 2a. IDs from test.csv, not gender_submission | yes | yes (+ overlap assert) | yes (+ called out in result.json) |
| 2b. Row counts 712/179 verified | no | yes (asserts) | partial (mentioned, not asserted) |
| 2c. Literal `PassengerId,Survived` header | yes | yes (re-read to confirm) | yes |
| 2d. Integer 0/1 predictions | yes | yes | yes |
| 2e. Majority + gender baselines reported | no | yes | yes |
| 2f. Missingness explicit + xx.5 Age flag | partial (no xx.5 flag) | yes (Age_estimated + Age_missing) | partial (no xx.5 flag) |
| 2g. No family-survival leakage across split | yes (none built) | yes (explicit) | yes (none built) |

## Verdict

**Ranking: C > B = A.** C edges out the other two on raw test accuracy
(+0.56pp; 1 extra correct row out of 179). A and B are dead-tied on the
leaderboard despite differing on 4 predictions internally. B is the cleanest
process-wise (audit 8/8) but the audit discipline did not buy it accuracy
here. A's vanilla pipeline performed indistinguishably from B's audit-driven
one — Titanic is too easy for the audit checklist to translate into a score
gap. The audit's main *honeytrap* (gender_submission PassengerIds) snagged
nobody, including the vanilla solver, so it does not discriminate on this run.
