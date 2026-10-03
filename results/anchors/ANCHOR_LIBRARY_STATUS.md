# Anchor library — status & executive decision (2026-06-19)

## Executive decision (made in the authors' absence): DO NOT auto-mutate effect values before the leaderboard

Per the project's standing rule (ledger D-019: no silent mutation of `META_BENCHMARKS` on single-pass
LLM reads) and the authors' provenance-first habit, I am **running the convergence leaderboard on the
current, documented anchor library** and **deferring all effect-value changes to review**. This is
reversible, reproducible, and keeps the headline number auditable. Corrections are staged below, not
applied.

What I DID apply (safe, provenance-only): nothing that changes a `meta_r` value. The suspect
abstract-step `0.31` cluster was already downgraded to `abstract_unverified_suspect` (D-019); that
stands.

## What is staged for review (apply on return, then re-run the anchoring leaderboard)

### Tier A — full-text VERIFIED, safe to adopt (4 + magnitudes)
From OA full text (`FULLTEXT_VERIFICATION.md`), sign + plausibility check out:
- `affect_valence → work_creativity`  0.15 (confirms claim)
- `social_energy_drain → work_engagement`  −0.19 (was −0.22)
- `social_energy_drain → psych_safety`  −0.45 (was −0.31, stronger)
- `social_energy_drain → work_creativity`  −0.236 (was −0.18, stronger)

### Tier B — corrected SOURCE attributions (25 pairs; fixes which paper, see RESOURCING_PROPOSALS.md)
The OpenAlex abstract-match mis-attributed 27 pairs (PRISMA guideline paper, a brain-lateralization
paper on 4 creativity pairs, robotics/music-education false hits, etc.). The re-sourcing agent found
the correct meta-analysis with a resolving DOI for 25 of them (e.g., Sio & Ormerod 2009 incubation =
10.1037/a0014212; the real affect→engagement ρ=.62 anchor = Young et al. 2018, 10.1002/job.2303).
Updating the citation/DOI improves provenance; whether each corrected source changes the `meta_r`
VALUE needs a human read of that paper's focal table. **Recommend: update citations now, re-verify the
focal r per paper before changing any value.**

### Tier C — QUARANTINED, do NOT adopt without a human PDF read
- `financial_worry → work_engagement`  full-text read flipped sign (−0.15 → +0.23). Theoretically
  dubious; almost certainly a reverse-coded / wrong-construct table cell. **Hold.**
- 4 creativity anchors stuck at exactly r = 0.31 (`*→work_creativity`, abstract-step artifacts,
  `abstract_unverified_suspect`). **Re-verify or drop.**

### Tier D — unresolved
- 2 pairs have no genuine meta-analytic source (narrative-review fallback only):
  `flow_hours → work_creativity`, `sleep_quality → work_creativity`.
- 49 paywalled anchors retain literature-claimed values (documented), pending the 36-paper manual
  worklist (`MANUAL_DOWNLOAD_WORKLIST.md`).

## Consequence for the headline figure
The anchoring leaderboard run now uses the FULL library (incl. low-confidence/inferred anchors). The
**high-confidence subset (19 high / 32 medium of 118)** should drive the headline; the full-library
number is a sensitivity check. Re-run `anchoring_leaderboard.py` after Tier A+B are applied and add a
high-confidence-only variant before camera-ready.
