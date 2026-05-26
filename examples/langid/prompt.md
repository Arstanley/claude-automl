# Example: Search-Query Language Identification

This is the v1 reference task for the AutoML harness. Paste this prompt
(or a shortened version) into Claude Code as `/automl <prompt>`.

---

The search team in charge of Adobe Express, Adobe Document Cloud,
Adobe Stock, Adobe Lightroom, and Adobe Creative Cloud Home wants to be
able to detect the languages in which a user search query is written.
Search queries are typically between 1 to 15 words, sometimes with typos,
sometimes agrammatical.

Your task is to create a language identification model that is as accurate
as possible. It should detect the languages below. Make sure model accuracy
is as good as possible on super short queries (1 to 5 words).

Languages to support:

- **Tier-1**: English, French, German, Japanese
- **Tier-2**: Spanish, Italian, Korean, Brazilian Portuguese
- **Tier-3**: Dutch, Swedish, Traditional Chinese, Simplified Chinese,
  Danish, Norwegian, Russian, Finnish, Polish, Turkish, Hungarian, Ukrainian
- **Additional**: Indonesian, Thai

Try to ensure that pure Hindi (e.g. "kaise ho") is NOT misclassified as
English (better to say "unsupported language" than to misclassify as English),
and explore whether en-IN (Indian English) detection is feasible (or at
least doesn't erroneously get classified as a different supported language).

If a language is not in the supported list, clearly detect it as
"unsupported language". Make sure it is not mistakenly recognized as one of
the supported languages.

Model constraints:
- Inference latency: below 100 ms on CPU
- Model size: max 10 GB

---

## What the harness will produce

- Run dir: `./automl_runs/<timestamp>-langid-.../`
- Structured constraints: 22 supported labels + `unsupported` + (optional) `en-IN`
- Attempts:
  - `fastText` supervised (likely winner under the tight latency budget)
  - `char-ngram + calibrated logistic regression` (tiny + interpretable baseline)
  - `distilled XLM-R-small` (test whether transformer accuracy is worth the cost)
- Eval slices:
  - `short_queries`: 1–5 words (primary)
  - `medium_queries`: 6–10 words
  - `long_queries`: 11–15 words
  - `romanized_hindi` (Hindi in Latin script)
  - `en_in_native` (Indian English samples)
  - `ood_languages` (languages NOT in the supported list — should predict `unsupported`)
- Constraint checks: per-attempt pass/fail on `max_latency_p99`, `max_model_size`,
  `hindi_as_english_fpr`, `macro_f1_target`
- Final artifacts: `report.md`, `model_card.md`, model files
