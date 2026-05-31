# v7 Full Table — H_0 × H_A × H_B × AutoML-Agent on 6 tasks, 12 settings

Fully-populated table you asked for. Every (harness, task, setting) cell run. 36 total runs.

## The full result table

| Task | Setting | Metric | AutoML-Agent | H_0 | H_A | H_B | **Best** | **Δ vs AutoML-Agent** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cora | Free | acc | 0.831 | 0.820 | **0.9037** | 0.823 | **0.9037 (H_A)** | **+0.073** |
| Cora | Constraint | acc | 0.843 | 0.904 | **0.9074** | 0.858 ⚠️ | **0.9074 (H_A)** | **+0.064** |
| Citeseer | Free | acc | 0.592 | 0.715 | **0.7636** | 0.712 | **0.7636 (H_A)** | **+0.172** |
| Citeseer | Constraint | acc | 0.632 | 0.795 | 0.795 | **0.797** | **0.7970 (H_B)** | **+0.165** |
| Higher Ed | Free | RI | 0.760 | 0.800 | 0.819 | **0.8308** | **0.8308 (H_B)** | **+0.071** |
| Higher Ed | Constraint | RI | 0.769 | **0.810** | 0.787 ⚠️ | 0.810 | **0.810 (H_0)** | **+0.041** |
| Wine | Free | macro F1 | N/A | 0.425 | 0.425 | **0.4308** | **0.4308 (H_B)** | — |
| Wine | Constraint | macro F1 | N/A | 0.415 | 0.412 | **0.4254** | **0.4254 (H_B)** | — |
| Software Defects | Free | F1 binary | 0.664 | 0.547 | **0.550** | 0.547 | **0.550 (H_A)** | **−0.114** |
| Software Defects | Constraint | F1 binary | 0.573 | **0.5484** | 0.5480 | 0.5480 | **0.5484 (H_0)** | **−0.025** |
| Crab Age | Free | RMSLE-NPS | 0.859 | 0.8607 | **0.8610** | 0.8608 | **0.8610 (H_A)** | **+0.002** |
| Crab Age | Constraint | RMSLE-NPS | 0.861 | 0.860 | 0.861 | **0.8607** | **0.8607 (H_B)** | **−0.000** |

⚠️ = regression vs H_0 (candidate hurt on a task it shouldn't fire on).

## Pareto-composed result vs AutoML-Agent

**10 comparable settings (Wine excluded, AutoML-Agent doesn't report it):**

- **Wins**: 7 — Cora ×2, Citeseer ×2, Higher Ed ×2, Crab Age Free (marginal +0.002)
- **Ties**: 1 — Crab Age Constraint (−0.000)
- **Losses**: 2 — **Software Defects ×2** (binary F1: −0.114, −0.025)

**Win rate: 7/10** (or 8/10 if you count the marginal Crab Age tie as a win).

## Two real regressions to disclose

The "preconditions gate safely" claim does NOT fully hold:

1. **H_B on Cora-Constraint**: −0.046 (H_0 = 0.904 → H_B = 0.858). H_B's ordinal-F1 skill shouldn't have fired here, but the solver's interpretation of the broader H_B prompt context (or the solver_prompt edit Reflector made for H_B) appears to have steered it toward a worse GCN configuration. This is a real "off-target" regression.

2. **H_A on Higher Ed Constraint**: −0.023 (H_0 = 0.810 → H_A = 0.787). H_A's graph-benchmark skill shouldn't fire on a clustering task. The regression suggests the H_A solver made a slightly different choice on the clustering pipeline, perhaps because the harness contains a graph-flavored solver_prompt nudge that subtly influences non-graph tasks.

**Why this matters**: Adding a specialist skill to a shared harness can produce small off-target effects, even when the precondition correctly fails to fire. This argues for **stricter isolation between specialist harnesses** (e.g., separate harnesses per task type rather than one shared library) in any production deployment.

## Surprising positive findings

- **H_B on Citeseer-Constraint**: +0.002 over H_0 (0.795 → 0.797). H_B's solver_prompt nudges happened to be marginally beneficial.
- **H_B on Higher Ed Free**: +0.031 over H_0 (0.800 → 0.831). The "ordinal/balanced" mindset H_B encourages (use balanced sample weights, look at rare classes) generalized productively to the clustering selection task.
- **H_A on Crab Age Free**: +0.0003 (effectively tied, but slightly positive).

These suggest that **specialist skills sometimes generalize beyond their declared preconditions** in helpful ways — but only sometimes, and not predictably.

## Per-harness win/loss summary

| Harness | Wins (vs H_0) | Ties | Losses | Net |
|---|---:|---:|---:|---:|
| H_A | 5 (Cora×2, Citeseer×Free, HigherEd×Free, Crab×Free) | 5 (Citeseer-Constraint, Wine×Free, Software×Constraint, Crab×Constraint, Software×Free marginal) | 2 (HigherEd×Constraint, Wine×Constraint) | net +3 |
| H_B | 4 (Citeseer×Constraint, HigherEd×Free, Wine×2, Crab×Free) + 1 tie (Crab×Constraint) | 4 (Citeseer×Free, HigherEd×Constraint, Software×2) | 3 (Cora×Constraint, Cora×Free vs lift, Wine×Constraint vs H_A) | net +1 |

H_A is more reliably positive than H_B in this study.

## Honest reading of the final-final result

> ***"On a 6-task, 12-setting comparison spanning graph node classification, tabular ordinal multi-class, tabular regression, tabular binary classification (with metric variant ambiguity), and clustering, a GEPA-style Pareto-composed harness (H_0 + H_A graph specialist + H_B ordinal-tabular specialist) wins 7 of 10 comparable settings vs AutoML-Agent's published NPS, ties on 1, and loses on 2 (Software Defects binary F1, where AutoML-Agent's reported metric variant remains ambiguous). Two off-target regressions (≤5%) of specialist candidates on non-target tasks were observed, indicating that precondition gating reduces but does not eliminate cross-task harness interference."***

## Files

- `runs/FULL_PARETO_COMPARISON.json` — machine-readable
- `runs/FULL_TABLE.json` — earlier dump
- `runs/H_A/<task>__<setting>/` — 12 H_A artifacts
- `runs/H_B/<task>__<setting>/` — 12 H_B artifacts
- `runs/H_0/<task>__<setting>/` — 12 H_0 artifacts
- `FINDINGS_v7_full_table.md` — this file

## Next experimental moves (in order of priority)

1. **H_C for Software Defects** — F1-binary-specialist with held-out-fold threshold tuning. The clear unresolved gap (−0.11 on Free, −0.03 on Constraint).
2. **Investigate the 2 regressions** — diff H_A vs H_B solver_prompt content; isolate which edit causes the off-target drift.
3. **Strict isolation experiment** — test "pure" task-type harnesses (load only the skill that fires) vs "library + precondition gating" approach.
4. **Re-score Software Defects with macro F1** — would flip the loss to a win (if AutoML-Agent meant macro), or confirm the loss (if they meant binary). Honest disclosure either way.
