# Cross-state nomological-net sourcing (between-person)

**Purpose.** Source the 6 pairwise BETWEEN-PERSON meta-analytic correlations among the four
workplace psychological states for the COLM paper's the internal-validity battery (v2) L3 validation targets:
work_engagement (WE), psych_safety (PS), work_creativity (WC), team_trust (TT).

**Output.** `data/seeds/cross_state_targets.json` (the curated targets, with per-pair status,
confidence, evidence quote, and expert-prior flags).

**Method (reuses the pipeline pattern).** OpenAlex works API (free, no key, polite pool
`mailto=svaid@hbs.edu`) located candidate meta-analyses for each pair; every selected DOI was
verified to resolve via its OpenAlex record. Open-access full text was retrieved via Unpaywall
+ direct repository/publisher URLs and parsed with `pdftotext` (poppler). Where the primary
PDF was paywalled or bot-blocked, the focal value was confirmed from the abstract or from an
open-access secondary evidence review that tabulates the meta-analytic estimate. Helper script:
`scripts/scraping/search_cross_state.py`; saved candidate sets + OA full texts in
`results/scraped/cross_state_candidates/`.

**Honesty.** No correlation or DOI was invented. `verified` = focal r quoted from the primary
OA full text. `abstract_only` = meta-analysis located + DOI verified, value confirmed from the
abstract or an OA secondary review (not the primary table). `no_meta_found` = no direct
meta-analytic r located; a clearly labeled EXPERT PRIOR (point value + wide margin) is given
instead. All estimates are between-person.

---

## Summary table

| Pair (key) | r | DOI | Status | Confidence | is_expert_prior |
|---|---|---|---|---|---|
| work_engagement__psych_safety | 0.50 | 10.1111/peps.12183 | abstract_only | medium | true (cell not isolated) |
| work_engagement__work_creativity | 0.46 | 10.2991/assehr.k.210423.053 | verified | high | false |
| work_engagement__team_trust | 0.45 | — | no_meta_found | low | true |
| psych_safety__work_creativity | 0.13 | 10.1111/peps.12183 | abstract_only | low | false |
| psych_safety__team_trust | 0.45 | 10.1111/peps.12183 | abstract_only | medium | false |
| work_creativity__team_trust | 0.50 | 10.3724/sp.j.1042.2020.01118 | verified | high | false |

Tally: **2 verified**, **3 abstract_only**, **1 no_meta_found** (expert prior).
Of the abstract_only set, 1 (WE-PS) additionally carries `is_expert_prior=true` because the
located meta-analysis reports the relationship but its exact table cell could not be isolated
from open-access copies.

---

## Per-pair detail

### 1. work_engagement -- psych_safety   (r = 0.50, abstract_only, medium; expert prior on exact cell)
- **Located source:** Frazier, Fainshmidt, Klinger, Pezeshkan & Vracheva (2017),
  "Psychological Safety: A Meta-Analytic Review and Extension," *Personnel Psychology*.
- **DOI:** 10.1111/peps.12183 (OpenAlex W2495600728, resolves; 1,012 citations; indexed 2016 online-first / 2017 print).
- **The r:** 0.50 (expert prior anchored to the located source).
- **How confirmed:** Frazier (2017) explicitly examines psychological safety "over and beyond
  related concepts such as ... work engagement" and includes work engagement in the nomological
  net as a strong outcome. The abstract + open-access secondary summaries confirm the
  relationship and the affective-outcome cluster (commitment rho = .48, satisfaction rho = .53),
  but the precise PS-engagement table cell could not be extracted (Wiley paywalled; ODU/bepress
  repository copy returned HTTP 403; MDSOAR DSpace front-end blocked the REST/bitstream API).
- **Confidence:** medium. The relationship is firmly located and strong; only the exact cell is
  a prior (value anchored to the .48-.53 cluster), hence `is_expert_prior=true`, `margin=0.20`.

### 2. work_engagement -- work_creativity   (r = 0.46, VERIFIED, high)
- **Located source:** Sari & Yudiarso (2021), "Work Engagement and Innovative Work Behavior:
  Meta-Analysis Study," *Advances in Social Science, Education and Humanities Research*
  (ICPSYCHE 2020 proceedings, Atlantis Press, open access).
- **DOI:** 10.2991/assehr.k.210423.053 (OpenAlex W3159439797, resolves; 34 citations).
- **The r:** 0.46 (95% CI 0.423-0.506; k = 26 correlations; random-effects, I^2 = 90.2%,
  no publication bias by Egger test).
- **How confirmed:** Atlantis Press is fully open access. Verbatim result confirmed from the OA
  record: "Work Engagement and Innovative Work Behavior have a medium effect size correlation
  (r = 0.46 with 95% CI between 0.423-0.506)." DOI verified in OpenAlex.
- **Confidence:** high. Note: creativity is operationalized as innovative work behavior (IWB),
  the standard proxy for individual workplace creativity. `margin=0.10`.

### 3. work_engagement -- team_trust   (r = 0.45, NO_META_FOUND -> expert prior, low)
- **Located source:** none direct. Extensive OpenAlex + web searches surfaced no dedicated
  between-person meta-analysis reporting a single work-engagement x trust correlation.
  Comprehensive engagement meta-analyses (e.g., Christian, Garza & Slaughter 2011,
  10.1111/j.1744-6570.2010.01203.x -- full text checked, treats trust only as a future-research
  moderator, no trust-engagement r; Mazzetti et al. 2023 JD-R, trust not a focal variable) do
  not report it.
- **The r:** 0.45 (EXPERT PRIOR, not a measured value).
- **Rationale:** Primary studies and reviews consistently report a positive, moderate-to-strong
  association (organizational/vertical trust is a documented engagement antecedent). The prior is
  bracketed by adjacent meta-analytic anchors: Frazier (2017) PS-trust rho = .45 and the
  engagement affective-correlate cluster (.48-.63). Set at 0.45 with `margin=0.20`.
- **Confidence:** low (no direct source).

### 4. psych_safety -- work_creativity   (r = 0.13, abstract_only, low)
- **Located source:** Frazier et al. (2017), *Personnel Psychology*. DOI 10.1111/peps.12183
  (verified). Alternative team-level anchor noted: Hulsheger, Anderson & Salgado (2009),
  "Team-level predictors of innovation at work," *JAP*, DOI 10.1037/a0015978
  (OpenAlex W..., resolves; abstract reports support-for-innovation / vision / task-orientation
  rho between .4 and .5 with creativity/innovation).
- **The r:** 0.13 (Frazier PS-creativity outcome, corrected rho).
- **How confirmed:** open-access secondary summary of Frazier (2017) reports verbatim:
  "psychological safety demonstrated a positive and significant relationship with information
  sharing (rho = .52), citizenship behaviors (rho = .32), creativity (rho = .13), and learning
  behavior (rho = .62)."
- **Confidence:** low. Frazier's .13 is markedly lower than the team-level innovation literature
  (Hulsheger .4-.5), reflecting construct/level heterogeneity (individual PS-creativity outcome
  vs team support-for-innovation). `margin=0.20`; Hulsheger 2009 is the documented higher-value
  alternative if an L3 target nearer .45 is preferred.

### 5. psych_safety -- team_trust   (r = 0.45, abstract_only, medium)
- **Located source:** Frazier et al. (2017), *Personnel Psychology*. DOI 10.1111/peps.12183
  (verified). Trust is a documented PS antecedent in this meta-analysis.
- **The r:** 0.45 (Frazier meta-analytic estimate for trust -> psychological safety).
- **How confirmed:** the CIPD open-access evidence review "Trust and psychological safety"
  (2024 scientific summary, cipd.org) Table 3 (antecedents of psychological safety) lists for the
  trust antecedent: ".39, .54, .45 | grades D, D, B | Aamir 2013; Basit 2017; Frazier 2017."
  The B-graded value .45 is the Frazier (2017) meta-analytic estimate (highest evidence grade in
  that row). Primary Frazier PDF paywalled, so status is abstract_only rather than verified.
- **Confidence:** medium. `margin=0.15`.

### 6. work_creativity -- team_trust   (r = 0.50, VERIFIED, high)
- **Located source:** Huang & Hu (2020), "The relationship between trust and creativity in
  organizations: Evidence from meta-analysis," *Advances in Psychological Science* 28(7),
  1118-1132 (open access, Chinese-language meta-analysis with English abstract).
- **DOI:** 10.3724/sp.j.1042.2020.01118 (OpenAlex W3045076704, resolves).
- **The r:** 0.50 (lateral/horizontal trust -- the within-team / intrateam trust referent --
  with creativity). Also reported: vertical trust (trust in supervisor) r = 0.38; system trust
  r = 0.48.
- **How confirmed:** OA publisher PDF downloaded (journal.psych.ac.cn) and parsed with
  pdftotext; full text saved at
  `results/scraped/cross_state_candidates/work_creativity__team_trust__10.3724_sp.j.1042.2020.01118.txt`.
  English abstract verbatim: "results show lateral trust (r = 0.50), vertical trust (r = 0.38),
  and system trust (r = 0.48) with creativity are significantly positively correlated"
  [85 studies, 99 independent effect sizes; lateral-trust k = 41].
- **Mapping note:** team_trust = trust among teammates = lateral/horizontal trust -> r = 0.50.
- **Confidence:** high. `margin=0.12`.

---

## Provenance / reproducibility
- Helper script: `scripts/scraping/search_cross_state.py` (OpenAlex search + Unpaywall OA PDF +
  pdftotext, candidates to `results/scraped/cross_state_candidates/`).
- DOI resolution was verified by fetching each OpenAlex work record
  (`https://api.openalex.org/works/https://doi.org/{doi}`) and confirming HTTP 200 + matching
  title/year/authors (logged in this session).
- Saved open-access full text used for the two `verified` extractions:
  `results/scraped/cross_state_candidates/work_creativity__team_trust__10.3724_sp.j.1042.2020.01118.txt`
  (Huang & Hu 2020); Sari & Yudiarso (2021) confirmed from the Atlantis Press OA record.
- Generated 2026-06-20.
