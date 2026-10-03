# The internal-validity battery of PsyDyBench (v2)

A ground-truth-free psychometric validation instrument for longitudinal synthetic populations.

## The contribution (what is new)
A synthetic population of psychological states lacks a held-out ground truth: the scores belong to
personas rather than to real people. the internal-validity battery (v2) turns "does this population behave like a human one?"
into a battery of **pre-registered equivalence tests against meta-analytic human benchmarks**, each with
**cluster-bootstrap uncertainty** and **multilevel (between/within) structure**. Three design commitments
carry the design:

1. **Equivalence tests.** Each criterion asks whether the synthetic statistic falls inside a
   pre-registered equivalence margin of a published human target (two one-sided tests, TOST, Lakens 2017). Verdicts are *equivalent / inconclusive / different*. A population earns
   credit only for equivalence it had the power to show.
2. **Anchored to the literature, with provenance.** Targets are taken from cited meta-analyses (norms.json for
   distributions and ICC, cross_state_targets.json for the nomological net), each tagged verified /
   abstract-only / expert-prior. A missing target yields `TARGET_UNAVAILABLE`,
   with the descriptive statistic and its CI reported in place of a verdict.
3. **Uncertainty by persona-cluster bootstrap.** Every statistic carries a 95% CI from resampling
   *personas* (the 6 day-rows are nested), so equivalence is judged on the full CI rather than a point estimate.

## Criterion → estimand → target → margin → test
| Level | Estimand | Target (source) | Margin | Test |
|---|---|---|---|---|
| **L1 location** | Hedges *g* of state mean vs norm | norms.json grand_mean/SD | *g* = 0.5 (Cohen 1988, "medium") | TOST on bootstrap CI of *g* |
| **L1 dispersion** | synth/target SD ratio | norms.json total_sd | [1/1.5, 1.5] (log) | ratio TOST |
| **L2 ICC** | ICC(1) via null multilevel model (REML) | norms.json icc (ESM range) | ±0.10 | TOST |
| **L2 within-SD** | pooled within-person SD ratio | total_sd·√(1−ICC) | [1/1.5, 1.5] | ratio TOST (primary variance discriminator) |
| **L3 between-r** | across-person-mean correlation, 6 state pairs | cross_state_targets (meta) | per-pair 0.10–0.20 | TOST per pair |
| **L3 within-r** | person-centered correlation, 6 pairs | none sourced | — | `TARGET_UNAVAILABLE` (r + CI) |
| **L4 dynamics** | within-person multilevel AR(1) fixed effect | none sourced | — | descriptive sign/direction (see caveat) |
| **L6 structure** | between- & within-level PC1 variance explained + eigenvalues | none | — | descriptive (honest 4-indicator approach) |
| **L7 range/ceiling** | scale coverage vs ±2SD expectation; floor/ceiling pileup | norms + 15% convention (McHorney & Tarlov 1995) | 15% | flag + bootstrap CI |

## Honest limitations (stated up front, not buried)
- **Multilevel eigenstructure in place of a 4-indicator CFA.** With only four state constructs a one-factor CFA has df = 2, and CFI/RMSEA are
  unstable and effectively unfalsifiable (Kline 2016). L6 therefore reports the multilevel eigenvalue
  structure (PC1 variance explained at the between and within levels) rather than fabricated fit indices.
- **AR(1) is biased at T = 6, and the battery treats it as directional.** A naive `state ~ lag` regression recovers
  ~+0.6 (between-person leakage). The correct within-person estimand is attenuated by the Nickell/Hurwicz
  dynamic-panel bias ≈ −(1+ρ)/(T−1) ≈ −0.26, so a planted within-person AR(1) of 0.30 reads as ~0.02. We
  report the within-person estimand (between-person-free) but flag L4 as **descriptive direction evidence
  only**, and draw the field-level implication: **validating longitudinal-persona *dynamics* (carryover,
  inertia) requires longer series than the 5 to 7 days typical of ESM-mimicking corpora.**
- **Target availability is partial.** Cross-state nomological targets: 2 verified, 3 abstract-only, 1
  expert-prior (quarantined, and the leaderboard reports a sourced-only equivalence rate excluding priors).
  Within-r, AR(1), and PC1 structure lack a published target and are reported descriptively.
- **The leaderboard ranks on continuous distances** (mean |g|, mean |ln within-SD ratio|, mean |Δr|),
  which are margin-free. Binary equivalence counts are secondary. So model *ordering* is independent of
  any threshold (verified by `threshold_sensitivity.py`).

## Reproduces
`scripts/validation/internal_validity_v2.py` (battery + `--selftest` recovering planted parameters),
`scripts/validation/run_validation_v2.py` (leaderboard_v2.csv with CIs), targets in
`results/leaderboard/internal_validity_v2_targets.json`, provenance in `results/anchors/CROSS_STATE_SOURCING.md`,
robustness in `scripts/analysis/threshold_sensitivity.py`.
