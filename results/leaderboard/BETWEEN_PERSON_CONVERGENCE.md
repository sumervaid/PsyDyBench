# Between-person convergence: the estimand-matched test (MAJOR CORRECTION, 2026-06-20)

## What changed
The published meta-analytic correlations (Frazier, Crawford, Dirks, Colquitt, …) are **between-person,
cross-sectional** effects (across people: "people with more autonomy report higher engagement"). The
convergence leaderboard previously reported **pooled** and **within-person** correlations — neither of
which matches that estimand. We added `compute_between_r` (correlate person-means across persons) and
re-ran. This is the like-for-like test.

## Result (28 models, full_v2 corpus; chance = 50% direction-match, rho = 0)
| level | mean direction-match | mean rho | matches meta_r estimand? |
|---|---|---|---|
| **BETWEEN-person** | **55.1%**  (range 46.6–61.9) | **0.10** (−0.1 to 0.3) | **YES** |
| pooled (blended) | 70.9% | — | no |
| within-person (day-to-day) | 73.4% | — | no |

**At the correct (between-person) level, meta-analytic convergence is at chance.** The earlier ~70%
"convergence" was the within/pooled level, which does not correspond to what the literature measured.

## Why: it is structural, by design
The seeder (`seed_personas.py::sample_indicators_for_day`) draws each day's 55 context indicators
**independently per day**, identical distribution for every persona, **with no link to the persona's
traits or baseline**. Verified: indicator between-person SD = 0.51 (pure sampling noise from averaging
6 IID draws, ≈ within_SD/√6) vs within SD = 1.34. So between persons the indicators carry **no
trait-linked signal** — there is nothing for the rater to recover at the between level, and between_r ≈
0 (random sign → ~50% direction-match). Confirmed across all 28 models, all capability tiers.

## Consequence for the paper
1. The meta-analytic **convergence framework, as currently seeded, does not demonstrate external
   validity at the between-person level** — it is at chance. The prior "CO-EQUAL, convergence cleared its
   bar" verdict was based on a mismatched estimand and is **retracted** pending the fix.
2. What the framework *does* weakly show is **within-person direction agreement** (73%, but with r ≈
   0.05 magnitudes — see the attenuation finding), i.e., the rater nudges the day's state in the
   literature-consistent direction. That is a within-person dynamics claim, not convergence with the
   (between-person) meta-analytic literature.
3. the internal-validity battery (internal structure) is unaffected — it never depended on the indicator→state anchors.

## The fix (trait-linked between-person seeding) — decision pending
To test convergence the way the meta-analyses measured it, re-seed so a persona's *average* indicator
level is partially tied to their traits/baseline (between-person structure), independent enough to avoid
circularity, then regenerate narratives and re-rate. This is a **full pipeline re-run** (re-seed →
regenerate days → re-rate all models): material compute + cost + time. It may still return ~chance if
the LLM does not couple between-person — which would itself be a clean, honest boundary finding.

Reproduce: `python scripts/analysis/anchoring_leaderboard.py results/generation/full_v2`
(`compute_between_r` in `scripts/validation/meta_analysis_validation.py`). Data:
`results/leaderboard/anchoring_leaderboard_3level.csv`.
