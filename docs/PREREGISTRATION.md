# Pre-Registration — persona-psychometrics-aims (Paper 1, COLM AIMS)

Fixed BEFORE any generation run. Timestamp anchored externally (OpenTimestamps `.ots` on this file's
hash; see `PREREGISTRATION.md.ots`). Edits after the anchor invalidate the pre-registration.

## Claim
Two ground-truth-free validation frameworks (GANDALF internal structure + meta-analytic convergence
anchoring external) can validate whether LLM-generated synthetic populations of daily psychological states
(WE/PS/WC/TT) are psychometrically real, demonstrated by an open-model benchmark.

## Model roster — FIXED (see `data/seeds/model_roster.csv`, 12 models)
Selection RULE (pre-specified, anti-cherry-pick): models chosen to span
(capability tier × family × size × dense/MoE × open/closed × reasoning/non), restricted to versions
available + version-pinned on OpenRouter (open) / OpenAI (closed), with within-family size pairs. The
12-model list is the output of this rule, not a post-hoc pick. Roster locked 2026-06-18.

## Design — FIXED
- Seeding: forward literature-parametric (published norms; `data/seeds/norms.md`), no real ESM.
- Generation: shared-event (narratives generated once, fixed model; each roster model RATES the same
  narratives), uniform rating prompt, no per-model DSPy.
- N = 300 personas × 6 days.
- States: WE(1–7), PS(1–4), WC(1–6), TT(1–5).

## Analysis plan — FIXED
- Per model: GANDALF 6 levels (pass/fail + score); meta-analytic anchoring (direction match %, signed
  Spearman ρ) vs the VERIFIED anchor library (`data/scraped/`); reliability (test–retest/split-half/
  G-theory); nomological (trait→state convergence).
- Primary output: leaderboard of which models yield psychometrically valid populations.
- Descriptive secondary: capability-vs-fidelity (order leaderboard by external capability score). NOT a
  causal RLHF claim (that is the follow-up paper).
- Cultural validation: OUT for this submission (hooks built; future work).

## What would falsify / surprise
- If all 12 models pass (framework non-discriminating) or all fail.
- If anchoring convergence does not exceed chance for any model.

## Budget guardrail
Halt if OpenRouter or OpenAI spend exceeds $50 (tracked separately, `scripts/budget_tracker.py`).
