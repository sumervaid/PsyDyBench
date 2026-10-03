# PsyDyBench data card

## What the data are
Synthetic, built entirely from published norms and meta-analytic effects. Every seed parameter traces to a published human dataset. The release ships these sets:

| Set | Location | Size | How produced |
|---|---|---|---|
| Persona seeds | `data/seeds/personas_full.csv` | 300 personas × 6 days | Sampled from published population norms (`data/seeds/norms.md`) by `scripts/seeding/seed_personas.py` |
| Shared stimuli | `results/generation/full_v2/_shared_events/` | 300 portraits + 1,800 day narratives | Generated once by a fixed language model from the seeds |
| Model ratings | `results/generation/full_v2/<MODEL>/states.csv` and `ratings.tar.gz` | 36 models × 1,800 persona-days | Each model's first-person end-of-day ratings of the shared stimuli |
| Cultural variant | `results/generation/cultural/` | 6 nations × 40 personas × 6 days, 3 models | Same pipeline with Hofstede-informed trait offsets (`seed_cultural.py`) |
| Anchor library | `scripts/validation/meta_analysis_validation.py`, `data/anchors/`, `data/seeds/*targets*.json`, `data/seeds/fulltext_extractions.json` | 118 indicator→state pairs | Effect sizes transcribed from published meta-analyses, with per-pair source audit |

## Intended use
Evaluating whether a language model, asked to inhabit a persona across consecutive days,
produces psychological-state trajectories whose statistical structure matches human
experience-sampling data and whose relations to context match the published literature.
The ratings are suitable for secondary analysis of model behaviour (e.g. studying the per-day
reasoning text in `ratings.tar.gz`). Their intended use is the evaluation of model behaviour. Substantive organizational research calls for human data.

## Excluded from the release, and why
- **Publisher PDFs** of the meta-analyses used to verify anchors are copyrighted and stay out of the release. Only the extracted values and their provenance are included (`docs/ANCHOR_SOURCES.md`).
- **API usage logs** written at run time (`results/budget/`) are excluded as operational noise.
- **Per-call API keys** are read from the environment (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`)
  and kept out of every file.

## Known limitations of the data
- Six days per persona. Over a panel this short the within-person autoregressive coefficient is
  biased, so the paper treats that criterion as directional.
- The stimulus corpus was written by one model. All 36 benchmarked models rate the same text, so
  the leaderboard isolates the rating stage. Compression already present in the narratives is
  shared by every model and therefore invisible to the ranking.
- Two of the 36 corpora are excluded from the paper's analyses as degenerate
  (`MISTRAL_SMALL_24B_INSTRUCT_2501`, a one-row corpus, and `QWEN3_5_9B`, a variance-error outlier).
  Their files are shipped unchanged.
- The 118 anchor effect sizes carry per-pair confidence tags. A subset was checked against the
  source paper's own tables where the full text was available (`results/anchors/FULLTEXT_EXTRACTED_VALUES.md`,
  `results/anchors/ANCHOR_LIBRARY_STATUS.md`). The rest were transcribed from abstracts or
  secondary sources. Convergence is reported both on the original library and on the
  re-anchored targets (`results/leaderboard/reanchored_convergence.csv`).

## Licensing and citation
Data: CC BY 4.0 (`data/LICENSE`). Code: MIT (`LICENSE`). Cite as in `CITATION.cff`.

## Provenance and integrity
`docs/PREREGISTRATION.md` fixed the roster, design, and analysis plan before any generation run
and is hash-anchored with OpenTimestamps (`docs/PREREGISTRATION.md.ots`, `docs/PREREG_HASHES.txt`).
`docs/DECISIONS.md` records every results-changing decision with its date and rationale.
