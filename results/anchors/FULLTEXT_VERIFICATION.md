# the pipeline · Step 4 — Open-access full-text verification of anchor effect sizes

**Date:** 2026-06-19 · **Script:** `scripts/scraping/fetch_fulltext.py` · **Host:** the compute host (`the compute host `), venv python
**Inputs:** `data/scraped/extractions/*.json` (118 anchor pairs) · **Extractor:** gpt-4.1-mini (temp 0), direct OpenAI
**Method:** DOI → Unpaywall (`api.unpaywall.org/v2/{doi}?email=svaid@hbs.edu`) → if OA PDF, download → pymupdf text (≤60k chars) → LLM-extract the focal meta-analytic *r* for the (indicator-construct, state-construct) pair → update extraction JSON.

## Why this step exists
The OpenAlex/abstract step (step 3) **located** the source meta-analysis for 113/118 pairs but could only read
abstracts — so for 107 pairs the focal correlation sat "in the PDF," unread. This step attempts a **legal,
automated** retrieval of the focal effect size from the open-access full text.

## Outcome (all 118 pairs, final status distribution)

| status | n | meaning |
|---|---|---|
| `verified`                     | 8  | focal *r* extracted; sign + magnitude agree with claimed (4 via full text, 4 via earlier abstract step) |
| `corrected`                    | 2  | focal *r* extracted from full text but **disagrees** with claimed (see below) |
| `oa_effect_not_found`          | 20 | OA full text read, but the pairwise meta-*r* is not reported in it |
| `oa_pdf_unreadable`            | 34 | Unpaywall gave an OA URL but it was a landing page / 403 / scanned image (no extractable text) |
| `source_found_effect_in_pdf`   | 49 | **no OA copy** (paywalled); effect remains in the closed PDF |
| `unfound`                      | 5  | no source located at all (step 3) |

**OA coverage:** Unpaywall returned a PDF URL for **60/111** DOI-bearing pairs; only **26** were machine-readable
and searched; of those, **6** yielded a confident focal *r* via full text.

**Honest limitation:** automated full-text effect extraction is bounded by (a) OA availability (~54% had any OA
URL; ~23% were actually readable), and (b) whether the *specific* pairwise correlation appears in running text vs.
a figure/correlation matrix the LLM could not parse. The remaining 49 paywalled pairs cannot be auto-verified;
they retain their literature-anchored claimed values and are documented as such.

## Full-text-verified pairs (`confidence_extraction = full_text`, this run)

| indicator → state | claimed *r* | full-text *r* | verdict |
|---|---|---|---|
| affect_valence → work_creativity | 0.15 | 0.15 | ✓ agree |
| social_energy_drain → work_engagement | −0.22 | −0.19 | ✓ agree |
| social_energy_drain → psych_safety | −0.31 | −0.45 | ✓ same sign, stronger |
| social_energy_drain → work_creativity | −0.18 | −0.236 | ✓ same sign, stronger |
| financial_worry → psych_safety | −0.12 | **−0.45** | ⚠ corrected (same sign, 3.75× stronger) |
| financial_worry → work_engagement | −0.15 | **+0.23** | ⚠⚠ corrected (**sign flip**) |

## ⚠ Flagged for human adjudication — NOT auto-applied to `META_BENCHMARKS`

Per project rule 3 (no ad-hoc results mutation) and basic measurement caution, single-pass LLM full-text reads were
**not** written into the benchmark library. Two items need a human eyeball before any update:

1. **`financial_worry → work_engagement` sign flip (−0.15 → +0.23).** Theoretically dubious (financial worry
   positively predicting engagement). Likely the LLM grabbed a reverse-coded or different-construct correlation
   from the table. **Do not adopt without checking the source PDF.**
2. **Four `verified` values stuck at exactly r = 0.31** (`support_received→creativity`,
   `competence_affirmation→creativity`, `affect_arousal→creativity`, `avg_energy_level→creativity`). These carry
   `confidence_extraction = "medium"` and `oa = None` — they are artifacts of the **earlier abstract step**, not
   this run (identical 0.31 across four distinct pairs = a parsing default). **These should be downgraded /
   re-verified, not trusted.**

The 4 clean full-text agreements + 1 strengthened magnitude (`social_energy_drain`, `affect_valence`) are safe to
adopt; `financial_worry` corrections await review.

## Provenance
- Updated audit records: `data/scraped/extractions/*.json` (in place; `claimed_meta_r` preserved alongside
  `verified_meta_r`, `oa`, `oa_pdf_url`, `evidence_quote`, `confidence_extraction`).
- Run log: `results/scraped/fetch_fulltext.log`.
- Token usage logged to `results/budget/openai_usage.jsonl` (gpt-4.1-mini; ~60 readable PDFs × ~1 call).
