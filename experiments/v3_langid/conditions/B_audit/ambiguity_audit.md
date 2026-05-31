# Ambiguity Audit — Language Identifier Task

**Spec under audit**: `/home/colligo/claude-automl/experiments/v3_langid/ambig_prompt.txt`
**Audit date**: 2026-05-28
**Stance**: adversarial. I am naming gaming surfaces and interpretive gaps the spec leaves wide open — not proposing a solution.

---

## 0. Data shape, established by inspection (not asserted by the spec)

- `papluca_{train,valid,test}.csv`: 70k / 10k / 10k rows, columns `labels,text`, **20 classes class-balanced** at every split (3500 / 500 / 500 per language). Label set: `ar bg de el en es fr hi it ja nl pl pt ru sw th tr ur vi zh`.
- **All 3,500 `hi` training rows are in Devanagari script**; zero of them are pure Latin. Verified.
- `hi_latn_dev.txt`: 5,000 lines, 99.2% ASCII-only, mean length 104 chars. Real romanized-Hindi sentences (e.g. `"iske aane se purva hi log gharon ki safai ka karya shuru kar dete hain."`). 419 lines are ≤20 chars.
- `hi_latn_test.txt`: 5,000 lines, same shape. Spec says **held out — do not train on**.
- Per-language ASCII-only fractions in train: `sw` 99.8%, `nl` 90.5%, `en` 86.9%, `it` 78.9%, `es` 28.4%, `de` 19.1%, `pt` 17.6%, `fr` 10.7%, `pl` 8.7%, `tr` 4.9%, `ur` 0.7%, everything else ~0%.
- Cross-split text duplicates exist: 112 train↔valid, 150 train↔test, 31 valid↔test. Minor but non-zero — affects "honest" test-set evaluation.
- Short rows are real: 206 train rows are ≤10 chars (e.g. `"وولف ."`, `"Çok iyi"`, `"Εγώ λίγο ."`). These are the natural place a model will fail silently.
- No nulls; column names are identical across splits; the `idx` for `submission.csv` will be 0..9999.

These facts ground every ambiguity below — the spec asserts none of them.

---

## 1. The headline ambiguity: romanized Hindi (the load-bearing trap)

The spec lists Hindi in the label set, but **every Hindi training sample is Devanagari**. `hi_latn_dev.txt` is offered as "available for your use" with zero guidance, and a held-out `hi_latn_test.txt` exists — strongly implying romanized Hindi is something the evaluator will probe.

A vanilla solver trains on `papluca_train.csv` and gets a model that has literally never seen Latin-script Hindi labeled as `hi`. When `hi_latn_test.txt` is presented:

- The 0–10ms character-ngram classifier will route almost all of it to **`en`** (87% ASCII overlap), **`nl`** (90% ASCII), **`sw`** (99% ASCII), `it`, or `tr` — whichever ASCII language won the bag-of-ngrams contest on training. The model will be **confidently wrong**, which the spec itself flags as the failure mode that "must not" happen.
- Even a solver who *notices* `hi_latn_dev.txt` has to choose between **at least four incompatible strategies**, each with different downstream consequences:
  1. **Ignore it.** Cleanest re: train/test discipline, but guarantees catastrophic failure on the held-out file.
  2. **Treat it as a 21st class `hi_latn`.** Now the label space disagrees with `papluca_test.csv`. The evaluator may score `submission.csv` against the 20-class label set and any `hi_latn` output is wrong by definition. Spec does not authorize new labels.
  3. **Add it to the `hi` training pool** (so Devanagari + Latin both → `hi`). Risks class imbalance (3500 Devanagari + 5000 Latin = `hi` is now the largest class) and pollutes the test distribution `papluca_test.csv` where `hi` is purely Devanagari.
  4. **Use it only for calibration / abstention thresholding** (e.g. "if top-1 prob < τ on Latin-script input, abstain"). Requires an abstention API the spec doesn't define.

**The spec does not pick one.** Every choice is defensible; every choice gives a different evaluator-facing number. This is the single largest gaming surface.

---

## 2. Latency and model-size numbers are missing

> "Latency and model size matter. You should care about p99 inference latency per query on CPU and total model size on disk. Don't ship a model so big it can't fit in our service budget."

The spec uses words like "p99", "service budget" — but **no numbers**:

- No latency target (1ms? 10ms? 100ms? 1s?).
- No model-size cap (10MB? 100MB? 500MB? 2GB?).
- No measurement protocol (cold start vs warm? single thread vs the predict.py harness? batched or one-at-a-time? text length stratification?).
- No tradeoff curve (is 99.5% acc at 200ms better than 98% acc at 5ms? we don't know).

A solver could ship anything from a 500KB char-ngram + linear-SVM (fast, ~98% on the 20 in-distribution classes, terrible on Latin-Hindi) up to a 500MB sentence-piece transformer-equivalent built from sklearn primitives (slow, possibly more robust). Both are "compliant" with the spec as written. The evaluator's actual threshold is opaque to the solver. **Gaming move**: optimize for accuracy and claim "size fits the budget" in `interpretation_notes` without numbers.

---

## 3. "Major user languages" is undefined

> "We care about the **major user languages** (the ones the search team cares about most), but the system shouldn't be hopelessly wrong on less common languages either."

The spec:

- Does not name the major-language tier.
- Does not say whether `hi` is in it (it's the literal subject of `hi_latn`, so probably yes — but the spec doesn't say).
- Does not say what metric weights major vs minor languages: macro-F1 (all equal), micro-F1 (proportional to test counts — irrelevant since test is balanced), weighted by some prior, or "tier-1 accuracy ≥ X, tier-2 ≥ Y".

If the solver trains an unweighted 20-class classifier, they are **implicitly declaring** all 20 languages equally important. If the evaluator later weights by `{en, es, fr, de, zh, ja, hi, ar, ru, pt}` (a plausible "search team" subset), the unweighted solver is silently underspending on the tier that matters. **There is no way for the solver to know without asking.**

Sub-ambiguity: "hopelessly wrong" on minor languages — what's the threshold? 30% accuracy? 70%? Random (5%)?

---

## 4. "Must not be confidently wrong" is undefined

> "There are some inputs the system **must not be confidently wrong** about — silently outputting the wrong language for these is worse than no answer."

This sentence implies an **abstention / refusal output**, but the deliverables don't define one:

- `predict.py` exposes `predict(text: str) -> str`. There is no documented sentinel for "unknown" or "refuse". Is `""`? `"unk"`? `"none"`? A label outside the 20? An exception?
- `submission.csv` schema is `idx,labels`. Does the evaluator accept non-label values in `labels`? Unknown.
- The set of "inputs" that trigger this rule is not enumerated. Obvious candidates:
  - Romanized Hindi (the screaming-obvious one).
  - Mixed-language / code-switched text.
  - Empty strings, whitespace-only, pure punctuation, single characters.
  - URLs, file paths, code snippets, JSON/XML.
  - Emoji-only or symbol-only inputs.
  - Latin-script inputs that aren't in the label set at all (Indonesian, Vietnamese-without-diacritics, Romanian, Catalan, …).
  - Numbers/dates/transliterated proper nouns ("Michael Kamen"-style).
- "Confidently wrong" implies a confidence threshold. Threshold of *what*? Softmax max prob? Top-1/top-2 margin? Calibrated probability? The spec gives no calibration target.

**Net effect**: the solver cannot know which inputs trigger the rule, what the abstention symbol is, or how confidence is measured. This is a second load-bearing gaming surface — a solver can claim "we abstain when prob < 0.6" with no way for the spec to disagree.

---

## 5. `hi_latn_dev.txt` is offered ambiguously

Restated from §1 for emphasis — the spec's exact phrasing is "**Available for your use**", which is among the most permissive and least directive instructions possible. Concrete consequences:

| Strategy | Pro | Con / risk |
|---|---|---|
| (a) Ignore it | Clean separation; no label-space confusion | Guarantees confident wrong-`en` on `hi_latn_test.txt` |
| (b) New class `hi_latn` | Honest separation of scripts | Breaks 20-class evaluation on `papluca_test.csv` |
| (c) Map to `hi` (add to training) | Single `hi` covers both scripts; matches `papluca_test.csv` schema | Pollutes test distribution; class imbalance; can hurt Devanagari `hi` accuracy on `papluca_test.csv` |
| (d) Calibration/abstention only | Preserves test distribution | Requires abstention contract the spec doesn't define |
| (e) Mix of (c)+(d) | Most defensible | Solver has to invent both decisions and document them |

The spec does not flag this choice as load-bearing, so a naive solver will silently pick (a) and pass `papluca_test.csv` but fail the implied romanized-Hindi probe.

---

## 6. Other gaming surfaces

- **Short-text degradation.** 206 train rows are ≤10 chars and 419 of the 5000 `hi_latn_dev` lines are ≤20 chars. The spec doesn't say how short inputs are scored or whether they're stratified separately. Char-ngram classifiers degrade hard below ~30 chars; a solver can quietly do nothing here.
- **Script-based heuristics.** Languages with disjoint scripts (Arabic, Greek, CJK, Thai, Cyrillic, Devanagari) are trivially separable by Unicode block — a 10-line rule beats any TF-IDF on those. The spec doesn't ban heuristics; a "clever" solver can shortcut accuracy with rules and bury the ambiguous Latin-script subset, hiding the real failure mode.
- **Latin-only confusables.** The hard subproblem is `{en, nl, sw, it, de, es, fr, pt, pl, tr, vi, hi_latn?}` — all share ASCII. Macro-F1 over 20 languages averages this hard subset with the trivial scripted ones, **inflating the headline number**. Spec doesn't ask for a Latin-only breakdown.
- **Train/valid/test text overlap.** 112+150+31 duplicate texts across splits is a real (small) leakage signal. Spec doesn't address dedup; a memorization-friendly model gets ~free points on the duplicates.
- **`papluca_valid.csv` purpose.** Spec mentions it exists but never says how it should be used (early stopping? hyperparameter tuning? threshold tuning? abstention calibration on Latin-script subset?). A solver can train on `train+valid` for a free accuracy bump unless prohibited — spec doesn't prohibit.
- **`submission.csv` ordering.** Spec says `idx` is "0-indexed row number into `papluca_test.csv`" — so the ordering should match the file. But spec doesn't say whether all 10,000 rows must be present, whether duplicate `idx` values are tolerated, or what happens if a row is missing or labelled with a non-20 token (see §4).
- **`predict.py` latency benchmark protocol.** Spec says it's "used by the evaluator to benchmark latency" but never specifies: batch size 1? text length distribution? warm vs cold? mean or p99? wall vs CPU? GIL contention? A model with heavy import-time work (loading a 100MB pickle) can game cold-start metrics either way.
- **Model artifact format.** `model.pkl` is sklearn pickle/joblib. Spec does not constrain what's inside — could be a pipeline, could be a dict of regexes, could be a 500MB compressed dictionary of memorized strings. There's no audit hook on the artifact.
- **"No fine-tuning of LLMs"** — explicitly forbidden, but the spec is silent on (i) using pretrained word/character embeddings shipped with sklearn ecosystem, (ii) using rule-based dictionaries / langdetect-style profiles, (iii) using compiled Unicode-block heuristics. Probably allowed, but a strict evaluator could call any of those "not in the spirit".
- **"No internet"** — pretrained resources downloaded before the run are technically not internet-at-runtime. Ambiguity remains.

---

## 7. Spirit-of-spec checklist

The spec rewards a solver who treats this as a production routing problem, not a leaderboard contest. A defensible attempt should:

- **Anticipate distribution shift.** `hi_latn_test.txt` exists because the evaluator expects to probe Latin-script Hindi. A solver that doesn't even acknowledge this gap in `interpretation_notes` is gaming.
- **Define an abstention contract explicitly.** Pick a sentinel (`"unk"`, `""`, or a confidence floor), document it in `result.json`, and apply it where confidence is low — particularly on Latin-script inputs that aren't decisively one of the Latin-script training languages.
- **State the (assumed) latency and size budgets numerically.** If the spec doesn't, the solver should declare e.g. "≤50ms p99 single-thread, ≤50MB on disk" in `constraint_choices` and stick to it. Silence here is the failure mode.
- **Name the major-language tier.** Pick a tier (e.g. UN-six + top Adobe markets), document it, and report per-tier metrics — not just macro-F1.
- **Report stratified metrics.** At minimum: per-language F1, accuracy on Latin-script subset, accuracy on short inputs (≤20 chars), and a Latin-Hindi confusion breakdown. A single headline number is gameable; stratified numbers force honesty.
- **Make the `hi_latn_dev.txt` decision explicit and reversible.** Whichever of (a)–(e) the solver picks (§5), it must be in `interpretation_notes` with the reasoning — not buried in code.
- **Respect held-out data.** Touching `hi_latn_test.txt` in any capacity (including reading line counts for "calibration") is out of bounds. Same for evaluating on `papluca_test.csv` to tune.
- **Don't pretend ASCII confusables are solved.** Headline macro-F1 will hide poor performance on the `{en,nl,sw,it,…}` cluster behind the trivial scripted languages. A solver who ignores this isn't building a search router; they're building a Unicode-block detector with a thin TF-IDF on top.

---

## 8. Summary of gaming surfaces, ranked

1. **Romanized Hindi (`hi_latn`)** — undefined label-mapping decision, no abstention contract, held-out test set strongly implies the evaluator will probe it. [§1, §5]
2. **"Must not be confidently wrong"** — no abstention symbol, no confidence definition, no enumerated trigger set. [§4]
3. **Latency / model-size budget** — no numbers, no measurement protocol. [§2]
4. **"Major user languages"** — tier undefined, weighting undefined. [§3]
5. **Latin-script confusables** — the actual hard subproblem, hidden under macro-averaging. [§6]
6. **Short-text behavior** — known degradation regime, no stratified scoring. [§6]
7. **`valid` split usage** — purpose undeclared, leakage into final training not prohibited. [§6]
8. **`predict.py` benchmark protocol** — measurement details opaque. [§6]

A solver who silently picks the "easy" interpretation on each axis (ignore `hi_latn_dev`, no abstention, no per-tier weighting, no stratified reporting) can produce a high macro-F1 on `papluca_test.csv` and still fail the routing problem the spec describes.
