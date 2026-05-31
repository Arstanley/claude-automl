# Eval — playground-series-s3e13 (MPA@3 / hit@3)

Gold target: `prognosis`. eval/eval.py implements hit@3 (true label anywhere in top-3 split-on-whitespace).

## Validity (all three: PASS)

| Condition       | rows | ids==test | header | tok/row | distinct | underscores | aligned-to-303 sample |
|-----------------|------|-----------|--------|---------|----------|-------------|------------------------|
| A_no_skills     | 142  | yes       | id,prognosis | 3 | yes | yes | no |
| A_with_skills   | 142  | yes       | id,prognosis | 3 | yes | yes | no |
| B_audit_round2  | 142  | yes       | id,prognosis | 3 | yes | yes | no |

No solver hit any of the three honeytraps (303-row sample, single-label, underscore-to-space).

## DSBench MPA@3 (hit@3) — true scores

| Condition       | score      |
|-----------------|------------|
| B_audit_round2  | **0.6197** |
| A_no_skills     | 0.5986     |
| A_with_skills   | 0.5775     |

## Headline-vs-true gap

- A_no_skills: claimed MAP@3 0.447 (rank-weighted, different metric); CV hit@3 reported 0.614 vs test 0.599 → +0.015 optimism.
- A_with_skills: claimed MPA@3 (hit@3) CV 0.628 vs test 0.577 → -0.051 optimism (modest, within CV std 0.031).
- B_audit_round2: claimed MAP@3 0.447 as headline; also reported hit@3 CV 0.621 vs test 0.620 → essentially zero gap. Best-calibrated solver.

## Audit-checklist hit rate

- A_no_skills: 5/7 (no explicit prior baseline / spec-vs-data row-count reconciliation).
- A_with_skills: 7/7 (skills_applied list cites schema-diff target, ignore-stale-sample, prior baseline 0.329, classes_ correctness, etc.).
- B_audit_round2: 7/7 with each audit item explicitly cross-referenced.

## v2 deltas

- **Transfer signal** = score(A_with_skills) − score(A_no_skills) = **-0.0211** (NEGATIVE).
- **Ceiling gap** = score(B_audit_round2) − score(A_with_skills) = **+0.0423**.

## Brief takeaways

- Negative transfer signal: the skill library nudged A toward a wider model grid (LR×3, NB×2, RF, ET, HGB×2, GBM); CV picked ExtraTrees(depth=8) as best-single, which overfit the n=565 / 11-class problem versus A_no_skills' simpler soft-vote LR+RF+BernoulliNB.
- Positive ceiling gap: B's audit pushed it to a more conservative LR+RF+ensemble grid with stronger CV calibration. RF won and its CV hit@3 (0.621) matched test (0.620) almost exactly.
- All three correctly defended `prognosis` as target and avoided the disjoint sample_submission.csv id range.
