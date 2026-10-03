# PsyDyBench method overview

This document is the reader's map from the paper's methods to the code and files in this
repository. The paper itself is the authoritative description. This page says where each piece
lives and what it reads and writes.

## Pipeline

```
published norms  ──►  seed personas  ──►  generate shared stimuli  ──►  rate (per model)  ──►  score
data/seeds/norms.*    scripts/seeding/      scripts/generation/          scripts/generation/      scripts/validation/
                      seed_personas.py      stage1_identity.py           stage3_rating.py         scripts/analysis/
                                            stage2_events.py
```

### 1. Seeding from published norms
`scripts/seeding/seed_personas.py` draws 300 synthetic personas. Big Five traits are drawn from a
multivariate normal whose means, standard deviations, and intercorrelations are the published
population values in `data/seeds/norms.md` / `norms.json`. Each persona's baseline on the four
states (work engagement, psychological safety, work creativity, team trust) is a linear function
of its traits using published trait→state coefficients. The within-person standard deviation of
each state is derived from published intraclass correlations. Daily context indicators are then
sampled forward for each of 6 days. The state columns stay unpopulated, so any recovered indicator-state relation is
emergent rather than seeded.
Output: `data/seeds/personas_full.csv` (1,800 persona-days).

### 2. Shared stimuli, generated once
`stage1_identity.py` writes a psychological portrait for each persona and `stage2_events.py`
writes a narrative for each persona-day from its sampled indicators. Both steps run once on a
single fixed model. The resulting 2,100 files (300 portraits + 1,800 day narratives) are the
benchmark's stimulus set and ship in `results/generation/full_v2/_shared_events/`. Every
benchmarked model reads exactly these files.

### 3. First-person rating, per model
`stage3_rating.py` places each benchmarked model inside the persona. The system message is
"You are the person described. Respond in first person, honestly." The prompt gives it the portrait, the
day's narrative, and that persona's own baseline and typical range for each state, and asks for
an end-of-day self-report of the four states at 5:30pm. The rating prompt is byte-identical across
models and is uniform across models (see the paper's appendix for the verbatim prompt).
`run_benchmark.py` orchestrates this concurrently and resumably across the roster in
`data/seeds/model_roster_scaling_live.csv`, routing open-weight models through OpenRouter and
OpenAI models through the OpenAI API. Output per model: `results/generation/full_v2/<MODEL>/states.csv`
(1,800 rows) plus the raw per-day responses in `ratings.tar.gz`.

### 4. Two scoring frameworks
**Internal validity.** Does the synthetic population reproduce the statistical structure of
repeated human measurement? `scripts/validation/internal_validity_v2.py` implements the six-criterion
battery (distributions, between/within variance partition, cross-state correlations, temporal
autocorrelation, latent dimensionality, scale-range use) as two one-sided equivalence tests
against published targets, with persona-cluster bootstrap intervals. `run_validation_v2.py`
drives it across models. `run_validation.py` produces the level pass/fail summary and the
within-person variance error used to rank models. See `docs/INTERNAL_VALIDITY.md` for the full
specification and the rationale for reporting two criteria in adjusted form.

**External validity.** Does the population recover relations established in the human
literature? `scripts/analysis/anchoring_leaderboard.py` correlates each day's seeded indicators
with each model's rated states (between and within persons) and scores direction agreement and
signed rank correlation against the 118 published meta-analytic effects in
`scripts/validation/meta_analysis_validation.py`. `scripts/analysis/nomological_scaling.py` runs
the trait→state check (do more conscientious personas report more engagement, etc.).

### 5. Secondary analyses
- `scripts/analysis/scaling_curves.py`: validity against log parameter count, by model family.
- `scripts/validation/validate_cultural.py` and `scripts/seeding/seed_cultural.py`: personas re-seeded
  for six nations with Hofstede-informed trait offsets. Measurement invariance of the cross-state
  structure and direction of cross-cultural differences.
- `scripts/magnitude/`: the appendix probe of why within-person correlations compress:
  re-rating reliability (`run_openai_reliability.py`, `analyze_disattenuation.py`), prompt
  interventions that remove the baseline anchor (`magnitude_search.py`, `run_single_variant.py`),
  low-ICC constructs (`new_states.py`, `rate_new_states.py`), and an all-silico MIPROv2
  achievability tune (`dspy_tune.py`).
- `scripts/scraping/`: the pipeline that located and, where open access allowed, verified the
  meta-analytic anchors (`docs/ANCHOR_SOURCES.md`).

### 6. Numbers in the paper
Every quantity the manuscript reports is a LaTeX macro in `results/paper/values.tex`.
`scripts/verify_paper_values.py` recomputes each one from the files above and reports the source.
`docs/PROVENANCE.md` holds the resulting table.
