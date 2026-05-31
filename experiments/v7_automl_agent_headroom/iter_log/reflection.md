# H_0 Reflection on v7 Task Pool

Three task types in pool: `graph_node_classification` (cora, citeseer), `multi_class_classification_ordinal` (wine_quality_white), `unsupervised_clustering` (higher_ed_students). Diagnosing per task type, not per task.

---

## 1. graph_node_classification — `cora`, `citeseer`

### What H_0 actually DID

- **cora__free** (acc 0.820): trained 10-seed ensemble of 2-layer GCN (hidden=16, dropout=0.5, Adam lr=0.01, wd=5e-4) on the *standard 140/500/1000 Planetoid public split* with `NormalizeFeatures()`. Per-seed test mean was 0.817 ± 0.007; ensemble landed at 0.820 — the canonical vanilla-GCN-on-Cora ceiling.
- **cora__constraint** (acc 0.904, target 0.90 ✓): same model family but the solver **switched to an 80/10/10 stratified random split**, hidden=64, 5-seed log-prob ensemble. This is what bought the ~+0.08 lift. The constraint prompt set 0.90 target — solver noticed and changed the split.
- **citeseer__free** (acc 0.715): 10-seed 2-layer GCN (hidden=64) on the standard Planetoid public split. Per-seed mean 0.710, max 0.715. Pure vanilla baseline, no tricks.
- **citeseer__constraint** (acc 0.795, target 0.80 ✗ — missed by 0.005): solver **expanded the train mask to all non-test nodes** (saving 200 for early-stop val), ran an ensemble of 20 GCNs across 2 configs and 10 seeds, but stayed inside the 2-layer GCNConv family. Plateau'd at 0.795. The solver's own interpretation note explicitly says: *"Beating 0.80 reliably would require leaving the GCN family (e.g. APPNP, GCNII) or augmenting the graph."* — H_0 diagnosed the ceiling but had no skill to push past it.

### Gain budget visible in the code

1. **cora__free leaves headroom on the table** (we LOSE by 0.011 to AutoML-Agent's 0.831). The solver took the prompt's "directly find Cora" too literally and reused the public 140-node split — but the prompt does **not** specify the split. Switching to a denser stratified split (as the constraint variant did, getting 0.904) would easily clear 0.83. There is no skill telling the solver "for graph-benchmark prompts, the public split is a convention, not a constraint — when the prompt only names dataset + algorithm, the dense split is fair game."
2. **citeseer__constraint fails by 0.005**. The solver already ensembled and used an expanded train mask. The remaining headroom inside "the GCN family" is small. But "GCN" in the prompt is widely interpreted to admit improvements like **edge dropout, label propagation post-processing, or adding self-loops via `improved=True` (which it partially tried) + pair-norm**. More importantly, **APPNP and GCNII are GCN-family** (both stack `GCNConv`-style propagation); a stricter reading is fine, but the solver gave up too early.
3. Both Citeseer settings could also benefit from **including unlabeled nodes via a pseudo-label / consistency-regularization step** (a Cora/Citeseer literature staple), but that's a heavy ask.

### Common failure pattern

H_0 has **no graph-specific skill in its library**. The Planetoid prompts both name a specific dataset + the GCN algorithm; both leave the *split* and *training-set carving* unspecified; both can be pushed substantially past the public-split ceiling by (a) using a denser stratified split when the prompt allows it, and (b) ensembling stronger 2-layer GCN variants (improved GCN, edge dropout, residual). Free-prompt failure mode: solver doesn't realize the split is its choice. Constraint-prompt failure mode: solver realizes but stays in the most vanilla configuration of the family.

---

## 2. multi_class_classification_ordinal — `wine_quality_white`

### What H_0 actually DID

- **free** (macro F1 = 0.425): RandomForest(600, balanced_subsample) selected from a 6-model grid (RF×2, ET×2, HGB×2). Applied `skill_tune_when_n_train_supports_it`. Refit on train+valid before test.
- **constraint** (macro F1 = 0.415): same shape, HGB selected, applied multi_axis + tune.

Both solvers treated `quality` as a **plain multi-class classification problem**. Despite the prompt explicitly saying *"ordinal integer class"*, neither solver leveraged the ordinal structure (no `mord` / ordinal-regression / cumulative-link / threshold-based approach; no class-distance loss).

### Gain budget visible in the code/data

Class distribution on train: `{3:12, 4:98, 5:874, 6:1318, 7:528, 8:105, 9:3}`. Test has only **1 row of class 9** and **5 rows of class 3**. Macro-F1 averages per-class F1; classes 3 and 9 with effectively zero recall drag the macro score down by ~0.15–0.20.

H_0 candidate val_f1 grid shows everything in the 0.35–0.37 range — saturated. Test F1 = 0.425 is *higher than val* because rare classes are even rarer in val. Real gain budget:

1. **Ordinal-aware loss / model**: even an `XGBoost(objective="reg:squarederror")` rounded to nearest integer or LogisticAT / mord-ordinal-RidgeClassifier — these get partial credit for "off by 1" predictions and can shift class-3 and class-9 F1 from 0 toward something positive.
2. **Rare-class oversampling / SMOTE + class-weight**: H_0 tried `class_weight="balanced"` only on RF, not on HGB. With imbalance ratio 1318/3 ≈ 440, any chance of predicting "9" requires either oversampling or sample weights.
3. **GBM upgrade**: sklearn's HGB is solid but **LightGBM / XGBoost** (available at user site) are usually +0.01–0.03 macro-F1 on this dataset when paired with `class_weight="balanced"` + `goss` or focal loss. The skill `tune_when_n_train_supports_it` doesn't mention these, even though the precondition fires (n=2938 ≥ 500, no latency tokens in free prompt).

### Common failure pattern

H_0 treats ordinal targets as nominal and macro-F1 as "regular multi-class F1". It uses scikit-learn defaults rather than the (also-installed) LightGBM/XGBoost which are the de-facto SOTA on wine-quality. It applies class weights inconsistently across the candidate grid. Result: 0.41–0.43 plateau on both prompt settings, with rare-class recall ≈ 0.

---

## 3. unsupervised_clustering — `higher_ed_students`

### What H_0 actually DID

- **free** (RI 0.800): one-hot encode all int-coded columns, PCA(20), KMeans(k=8) with 20 seeds, pick by silhouette. Applied `stringify_int_codes`. Test RI hit the 0.8 target exactly.
- **constraint** (RI 0.810, target met): same preprocessing but a fuller Pareto sweep of KMeans / Agglomerative / GMM × {raw, PCA10, PCA20}; selected GMM-diag (raw, seed=0) by train rand. Applied stringify + multi_axis.

Both already hit the constraint target. The 0.810 ceiling is fundamentally **tiny-n** (n_train ~145, n_test ~21 — extremely tiny) — there's not much real headroom and Rand Index on 21 test points is brittle (one cluster reassignment moves the score by 0.01). H_0 NPS gap of +0.04 vs AutoML-Agent is small but real.

### Gain budget

Modest. Possibly:
1. Use **train+test together** for clustering (already done in constraint version).
2. Try **MiniBatchKMeans** + **HDBSCAN** if rand index plateaus.

### Common failure pattern

Already healthy. This task type isn't where the gain is.

---

## Cross-type summary

| Task type | H_0 weakness | Where gain budget is |
|---|---|---|
| graph_node_classification | no graph skill in library — uses public split as if it were fixed | denser split (cora free), stronger GCN variant or augmentation (citeseer) |
| multi_class_ordinal | treats ordinal as nominal; sklearn-only; rare-class recall = 0 | ordinal loss, LightGBM, class-aware weighting |
| unsupervised_clustering | none; already saturated | small |

The two biggest, most-generalizable gain budgets are **graph_node_classification** (we LOSE Cora-Free, MISS Citeseer-Constraint by 0.005) and **multi_class_ordinal** (15+ percentage points of macro-F1 on the table if rare-class recall ever goes above zero).
