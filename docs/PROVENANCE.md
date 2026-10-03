# Provenance of every reported number in PsyDyBench

Each quantity reported in the manuscript is a LaTeX macro
defined in `results/paper/values.tex`, and `scripts/verify_paper_values.py` recomputes each
macro from the files in this repository, compares it at the precision the paper prints, and
names the source. The table below is written by that script (`reproduce.sh` runs it last).

Status codes: **OK**, the recomputed value matches the paper at printed precision.
**UNUSED**, the macro is defined but the camera-ready manuscript leaves it unused.
A **MISMATCH** fails `reproduce.sh`.

Two conventions the table reflects:
- The paper's analysed roster is the 36 benchmarked models minus two degenerate corpora
  (`MISTRAL_SMALL_24B_INSTRUCT_2501`, one row, and `QWEN3_5_9B`, a variance-error outlier), giving 34.
  Level pass counts and the median variance error use this roster.
- The indicator→state convergence means are taken over the full anchoring file, as published.

<!-- BEGIN AUTO-TABLE -->
| macro | value in paper | recomputed | source | status |
|---|---|---|---|---|
| `\nModels` | `36` | 36.000 | `results/leaderboard/leaderboard.csv` | OK |
| `\nAnalyzed` | `34` | 34.000 | `results/leaderboard/leaderboard.csv minus the two degenerate corpora` | OK |
| `\nPersonas` | `300` | 300.000 | `results/leaderboard/leaderboard_v2.csv` | OK |
| `\nDays` | `6` | 6.000 | `results/leaderboard/leaderboard_v2.csv` | OK |
| `\nStates` | `4` | 4.000 | `scripts/generation/gen_config.py (OUTCOMES)` | OK (unused in manuscript) |
| `\nDevelopers` | `nine` | 9.000 | `results/leaderboard/nomological_leaderboard.csv (model prefixes)` | OK |
| `\paramLo` | `1B` | 1B | `results/leaderboard/nomological_leaderboard.csv` | OK |
| `\paramHi` | `671B` | 671B | `results/leaderboard/nomological_leaderboard.csv` | OK |
| `\nomoDir` | `84.3\%` | 84.265 | `results/leaderboard/nomological_leaderboard.csv` | OK |
| `\nomoRangeLo` | `75\%` | 75.000 | `results/leaderboard/nomological_leaderboard.csv` | OK |
| `\nomoRangeHi` | `100\%` | 100.000 | `results/leaderboard/nomological_leaderboard.csv` | OK |
| `\betweenConv` | `55.5\%` | 55.517 | `results/leaderboard/anchoring_leaderboard.csv (mean over all rows)` | OK |
| `\withinConv` | `72.6\%` | 72.589 | `results/leaderboard/anchoring_leaderboard.csv (mean over all rows)` | OK |
| `\bestVarFid` | `0.043` | 0.043 | `results/leaderboard/leaderboard.csv` | OK |
| `\bestVarFidModel` | `Gemma-2-27B` | Gemma-2-27B | `results/leaderboard/leaderboard.csv` | OK |
| `\gptFiveVarFid` | `0.061` | 0.061 | `results/leaderboard/leaderboard.csv` | OK |
| `\gemmaFourVarFid` | `0.427` | 0.427 | `results/leaderboard/leaderboard.csv` | OK |
| `\gemmaFourConv` | `74.6\%` | 74.600 | `results/leaderboard/anchoring_leaderboard.csv` | OK |
| `\nLevels` | `6` | 6.000 | `scripts/validation/internal_validity_v2.py (L1,L2,L3,L4,L6,L7)` | OK |
| `\lTwoPass` | `26` | 26.000 | `results/leaderboard/leaderboard.csv` | OK |
| `\lSevenPass` | `34` | 34.000 | `results/leaderboard/leaderboard.csv` | OK |
| `\lFourPass` | `7` | 7.000 | `results/leaderboard/leaderboard.csv` | OK (unused in manuscript) |
| `\maxLevelsPassed` | `3` | 3.000 | `results/leaderboard/leaderboard.csv` | OK |
| `\varErrMedian` | `0.208` | 0.208 | `results/leaderboard/leaderboard.csv` | OK (unused in manuscript) |
| `\nAnchors` | `118` | 118.000 | `scripts/validation/meta_analysis_validation.py` | OK (unused in manuscript) |
| `\anchorsVerified` | `10` | — | `` | UNUSED |
| `\cultMinInvar` | `0.014` | 0.014 | `results/leaderboard/cultural_validation.csv` | OK |
| `\cultMaxInvar` | `0.038` | 0.038 | `results/leaderboard/cultural_validation.csv` | OK |
| `\cultConv` | `75\%` | 75.000 | `results/leaderboard/cultural_validation.csv` | OK |
| `\relYMean` | `0.97` | 0.970 | `results/magnitude/ratings_GPT_4_1.csv (ICC(1,1) over re-ratings, T=0.8)` | OK |
| `\withinRbaseline` | `0.023` | 0.023 | `results/magnitude/pairs_V0_baseline_full.csv` | OK |
| `\nObsFull` | `1{,}800` | 1800.000 | `results/magnitude/search_summary.csv` | OK |
| `\dirFull` | `71\%` | 71.186 | `results/magnitude/pairs_V0_baseline_full.csv` | OK |
| `\rhoFull` | `0.38` | 0.379 | `results/magnitude/pairs_V0_baseline_full.csv` | OK |
| `\dirDeanchor` | `77\%` | 77.119 | `results/magnitude/search_summary.csv` | OK |
| `\rhoDeanchor` | `0.46` | 0.459 | `results/magnitude/search_summary.csv` | OK |
| `\withinRtempEight` | `0.059` | 0.059 | `results/magnitude/ratings_GPT_4_1.csv (T=0.8) via scripts/magnitude/eval_within_r.py` | OK |
| `\withinRtempZero` | `0.060` | 0.060 | `results/magnitude/ratings_GPT_4_1.csv (T=0.0) via scripts/magnitude/eval_within_r.py` | OK |
| `\miniRho` | `0.45` | 0.448 | `results/magnitude/pairs_V0_baseline_GPT_4_1_MINI.csv` | OK |
| `\nanoRho` | `0.48` | 0.481 | `results/magnitude/pairs_V0_baseline_GPT_4_1_NANO.csv` | OK |
| `\nanoDir` | `75\%` | 74.576 | `results/magnitude/pairs_V0_baseline_GPT_4_1_NANO.csv` | OK |
| `\exhaustICC` | `0.34` | 0.340 | `scripts/magnitude/new_states.py (literature ICC)` | OK |
| `\exhaustWithinR` | `0.058` | 0.058 | `results/magnitude/newstates_within_r.csv` | OK |
| `\exhaustDir` | `100\%` | 100.000 | `results/magnitude/newstates_within_r.csv` | OK |
| `\fatigueICC` | `0.40` | 0.400 | `scripts/magnitude/new_states.py (literature ICC)` | OK |
| `\fatigueWithinR` | `0.075` | 0.075 | `results/magnitude/newstates_within_r.csv` | OK |
| `\fatigueDir` | `100\%` | 100.000 | `results/magnitude/newstates_within_r.csv` | OK |
| `\fatigueRho` | `0.95` | 0.952 | `results/magnitude/newstates_within_r.csv` | OK |
| `\flowICC` | `0.48` | 0.480 | `scripts/magnitude/new_states.py (literature ICC)` | OK |
| `\flowWithinR` | `0.069` | 0.069 | `results/magnitude/newstates_within_r.csv` | OK |
| `\nIccPairs` | `8` | 8.000 | `results/magnitude/newstates_within_r.csv` | OK |
| `\miniBaseComp` | `0.693` | 0.693 | `results/magnitude/dspy_tune_gpt_4_1_mini.json` | OK |
| `\miniTunedComp` | `0.702` | 0.702 | `results/magnitude/dspy_tune_gpt_4_1_mini.json` | OK |
| `\nanoBaseComp` | `0.660` | 0.660 | `results/magnitude/dspy_tune_gpt_4_1_nano.json` | OK |
| `\nanoTunedComp` | `0.673` | 0.673 | `results/magnitude/dspy_tune_gpt_4_1_nano.json` | OK |
| `\nTuneTrain` | `30` | 30.000 | `results/magnitude/dspy_tune_gpt_4_1_nano.json` | OK |
| `\nTuneTest` | `30` | 30.000 | `results/magnitude/dspy_tune_gpt_4_1_nano.json` | OK |
| `\nVerifiedL` | `5` | 5.000 | `data/seeds/cross_state_targets.json (pairs with a located source)` | OK |
<!-- END AUTO-TABLE -->

## Regenerated reports

The per-model internal-validity reports (`results/leaderboard/<MODEL>_internal_validity_v2.json`) and
`leaderboard_v2.csv` in this release were regenerated on 2026-10-02 from the released corpora with the
pinned environment in `requirements.txt`. They differ from the June run in three ways. The
engagement and psychological-safety cross-state target was re-anchored on 2026-06-22 from an expert prior
(0.50) to the full-text value (0.45, margin 0.10), which changes the L3 equivalence counts. Two corpora
(`QWEN3_5_27B`, `QWEN3_5_9B`) were completed after the June leaderboard. Between-person standard
deviations from the mixed-model fit move in the third decimal. The manuscript macros that read these
files (`\nPersonas`, `\nDays`) are unchanged, and every macro in the table above was verified against
the regenerated files.

