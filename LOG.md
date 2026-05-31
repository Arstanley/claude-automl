# Research Log — claude-automl experiments

Chronological log of all experiments run in `experiments/`. The headline result is **v7**: a GEPA-style Pareto-composed harness wins 7 of 10 settings vs ICML 2025's AutoML-Agent on the tasks AutoML-Agent's paper reports. Detailed findings linked per round.

## Quick links

- **🏆 [v7 Full Table](experiments/v7_automl_agent_headroom/FINDINGS_v7_full_table.md)** — 36 runs, 6 tasks × 2 settings, H_0 × H_A × H_B comparison vs AutoML-Agent. The final headline.
- **[v7 Headline (Phase B)](experiments/v7_automl_agent_headroom/HEADLINE_RESULT.md)** — H_0 alone vs AutoML-Agent: 5/6 wins.
- **[v7 GEPA Final](experiments/v7_automl_agent_headroom/FINDINGS_v7_final.md)** — Pareto-composed (H_0 + H_A + H_B): 6/6 on original tasks.
- **[v7 Extended](experiments/v7_automl_agent_headroom/FINDINGS_v7_extended.md)** — added Software Defects + Crab Age (10 settings).
- **[v3 Adobe langid](experiments/v3_langid/FINDINGS_v3.md)** — multi-axis production-shape AutoML; vanilla harness silently fails Hi-Latn (0% → 98.7%).
- **[Next Steps](experiments/NEXT_STEPS.md)** — staged GEPA refinement idea (deferred).

## Timeline

| Round | What it tested | Key finding |
|---|---|---|
| v0 | Single-task spec-gaming (Adult Income) | Adversarial-evaluator framing changes solver's reporting honesty (gap −0.094 → −0.023). Anecdote. |
| v1 | 5 DSBench Ambig tasks, audit vs vanilla | Audit beats vanilla on 3/5 true test, mean +0.92 pp. First evidence intervention changes *model* quality, not just reporting. |
| v2 | 2 fresh DSBench tasks, skill-library transfer | Mixed: +0.015 QWK on s3e5, −0.021 hit@3 on s3e13. Skills can transfer or misfire. → Motivates conditional preconditions. |
| v3 | Adobe langid (multi-axis: latency + size + hard-rule + tiers) | **A_vanilla: 0% on Hi-Latn probe. B_audit/C_warned: 98.7%+ correct.** Strongest single-task result of the project. |
| v4 | Distill v1+v3 failures → v4 skill library | 7 conditional skills with mechanical preconditions. Anti-overfit guards (lesson from v2 misfires). |
| v5 | First harness-optimization attempt: per-bundle SkillOpt-style | 20 DSBench tabular tasks, single-shot. Mean train +0.06%, val +1.46% (outlier-driven), test −0.17%. OOF-threshold edit shipped and overfit. |
| v6 | Per-edit gating + larger pool (20→30 tasks) | Per-edit gating filtered the bad edits; net improvement small but robust. Training signal exists but is weak (median +0.20% on regression tasks). |
| **v7** | **Head-to-head vs AutoML-Agent (ICML'25 published numbers)** | **H_0 alone wins 5/6 settings; GEPA-style Pareto-composed wins 7/10 on extended set. Two regressions observed.** |

## The publishable claim, as currently demonstrated

> *On 10 settings spanning AutoML-Agent's main benchmark (6 tasks across graph node classification, tabular classification F1, tabular regression RMSLE, clustering RI, and multi-class ordinal F1), a Pareto-composed harness — Claude Opus 4.7 with a minimal v4 skill library plus two GEPA-style task-type specialists — wins 7 settings, ties 1, and loses 2 (both Software Defects binary F1, where AutoML-Agent's exact F1 variant is ambiguous) against AutoML-Agent's published NPS. The harness optimization itself contributes +0.084 on Cora-Free (flipping a 0.011 loss to a 0.073 win), +0.049 on Citeseer-Free, and +0.031 on Higher-Ed-Free. Two off-target regressions (≤5%) on non-target tasks indicate that mechanical preconditions reduce but do not eliminate cross-task harness interference.*

## Per-experiment cost

- v0: ~10 subagent calls
- v1: ~25 subagent calls
- v2: ~10 subagent calls
- v3: ~10 subagent calls
- v4: 1 subagent call (Reflector)
- v5: ~25 subagent calls
- v6: ~50 subagent calls (per-edit gating + 10 new train baselines)
- v7: ~50 subagent calls (8 H_0 + 12 H_A + 12 H_B + 1 Reflector + 4 new H_0 for Software Defects + Crab Age)

Total across this project: **~180 subagent runs**.

## What's not in this repo

- **Source datasets**: papluca/language-identification (HF), DSBench (74-task corpus from HF), Dakshina v1 (Google Research), UCI Wine Quality, UCI Higher Ed Students (ucimlrepo), Cora & Citeseer (PyG). All re-acquirable; download instructions in each task's setup.
- **Cloned upstream repos**: `experiments/{DSBench,DSAgent,AutoMLAgent}/` are upstream's own git histories; not vendored here.
- **Large model artifacts**: `.pkl` / `.pt` files (model checkpoints). Solver scripts (`solution.py`) regenerate them.
- **Secrets**: `/home/colligo/.secrets/azure_openai.env` — outside the repo, gitignored category-wise.

## How to read this repo

```
LOG.md                          ← this file
SKILL.md                        ← the /automl Claude Code skill (orchestrator)
agents/automl-*.md              ← 6 subagent definitions (planner, researcher, etc.)
webui/server.py                 ← FastAPI dashboard
schemas/                        ← state.json blackboard schema

experiments/
├── v0_spec_gaming/             ← Adult Income MVP
├── v1_dsb_target_ambig/        ← 5 DSBench tasks, audit vs vanilla
├── v2_skill_transfer/          ← Skill library cross-task test
├── v3_langid/                  ← Adobe langid multi-axis
├── v4_self_improve/            ← Conditional skill distillation
├── v5_harness_opt/             ← First harness optimization (per-bundle gating)
│   ├── harness/                ← H_0 (the initial harness used in v5/v6/v7)
│   ├── harness_H_1*            ← v5/v6 candidates (per-bundle then per-edit)
│   └── tasks/*                 ← 30 DSBench-tabular tasks (data gitignored)
├── v6_*                        ← Files folded into v5_harness_opt/ (per-edit gating outputs)
├── v7_automl_agent_headroom/   ← 🏆 The head-to-head result
│   ├── tasks/                  ← 6 acquired AutoML-Agent tasks (data gitignored, prompts kept)
│   ├── gepa_candidates/H_A     ← graph-specialist variant
│   ├── gepa_candidates/H_B     ← ordinal-tabular-specialist variant
│   ├── runs/H_0/{task}__{setting}/  ← H_0 outputs (solution.py + result.json kept)
│   ├── runs/H_A/{task}__{setting}/  ← H_A outputs
│   ├── runs/H_B/{task}__{setting}/  ← H_B outputs
│   ├── FINDINGS_v7_*.md        ← writeups for each phase
│   ├── HEADLINE_RESULT.{md,json}
│   └── FULL_PARETO_COMPARISON.json
└── NEXT_STEPS.md               ← staged GEPA-refinement direction (deferred)
```

## How to reproduce

For any experiment, e.g. v7 langid-from-scratch:
1. Re-acquire the 6 task datasets per their per-task `data/` setup (TSLib/HF/UCI/PyG as noted).
2. Spawn Claude Code subagents per the prompt templates in `experiments/v7_automl_agent_headroom/tasks/*/prompt_{free,constraint}.txt`.
3. Each subagent runs the corresponding `solution.py` under the chosen harness (`experiments/v5_harness_opt/harness/` for H_0, `experiments/v7_automl_agent_headroom/gepa_candidates/H_A/` or `H_B/` for the specialists).
4. Score `submission.csv` against held-out labels per each task's metric.
5. Compare to AutoML-Agent's Table 7 NPS.

## What still needs doing for an ICLR-quality submission

1. **N is still small.** 6 tasks → 12 settings. Extending to AutoML-Agent's other 8 main tasks (Banana, Butterfly, Shopee, Ecomm, Entail, Smoker-cluster, Crop, Weather, Electricity) needs Kaggle credentials we lack — would 10-15 hours with credentials.
2. **Single-run scores.** AutoML-Agent reports mean ± std over 5 runs. We don't yet.
3. **H_C for Software Defects.** The single unresolved gap. A careful held-out-fold F1-threshold-tuning skill should close it.
4. **Staged GEPA refinement** (per `NEXT_STEPS.md`): refine per pipeline stage (planning / data / execution / verification) instead of single global Optimizer. Future work.
5. **AutoML-Agent head-to-head with their exact code.** We compare against their *published numbers* (Table 7 NPS), not against running their code on our infra. A true controlled comparison needs their Azure setup + dataset replication — see `experiments/AutoMLAgent/configs.py` for the wiring we prepped but couldn't run (no Kaggle creds, Azure firewall).

## Acknowledgments

This work used:
- [AutoML-Agent (Trirat et al., ICML 2025)](https://arxiv.org/abs/2410.02958) as the published baseline.
- [DSBench (Jing et al., ICLR 2025)](https://arxiv.org/abs/2409.07703) as the primary training task corpus (v5/v6).
- [DS-Agent (Guo et al., 2024)](https://arxiv.org/abs/2402.17453) as the source for several task descriptions.
- [Dakshina v1 (Roark et al., 2020)](https://arxiv.org/abs/2007.01018) for romanized-Hindi adversarial probes (v3).
- [SkillOpt (2605.23904)](https://arxiv.org/abs/2605.23904) — per-edit validation-gated optimization, our v5/v6 protocol.
- [GEPA (Agrawal et al., 2025)](https://arxiv.org/abs/2507.19457) — reflective + Pareto-frontier optimization, our v7 protocol.
- [Trace2Skill (2603.25158)](https://arxiv.org/abs/2603.25158) — trajectory-distilled skills (v4 inspiration).
