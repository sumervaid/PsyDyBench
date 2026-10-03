#!/usr/bin/env python3
"""
Nomological-network validation (Framework 2, external validity — §4 of outline).

Trait -> state convergence. Where `META_BENCHMARKS` (in meta_analysis_validation.py)
anchors DAILY-CONTEXT indicators to states, this module anchors stable BIG FIVE
traits to the four states against published meta-analytic / large-sample effects.
Traits are the seeded persona inputs (stable, not rated), so trait->state effects
test whether the generated populations reproduce the established personality
nomological network — a between-person convergence check that complements the
indicator-level anchoring.

NOMO_BENCHMARKS pattern mirrors META_BENCHMARKS: one dict per (trait, state) pair
with an expected_r, a citation, a construct_match, and a confidence tier. Big Five
x 4 states. Values are meta-analytic where a direct construct match exists, else
inferred from the nearest established path (flagged lower confidence). As with
META_BENCHMARKS these should be DOI-verified before the camera-ready (the anchor
the anchor-verification pipeline); current values are literature-sourced and tier-flagged.

The tester joins per-person trait scores to per-person state means in a generated
corpus and reports, per pair: observed Spearman r, direction match vs expected,
and overall direction-match % + signed rho (the convergence statistic).

INPUT (--corpus): a CSV with one or more rows per employee containing the four
*_state columns AND Big Five trait columns. Trait column names are auto-detected
from a set of common aliases (e.g. extraversion / bfi_extraversion / E). States
are averaged to a person mean before correlating with the (constant) trait.

    python3 scripts/validation/nomological.py --corpus corpus.csv --out-dir results/nomological/<run>/
    python3 scripts/validation/nomological.py --demo --out-dir results/nomological/demo/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STATE_COLS = [
    "work_engagement_state",
    "psych_safety_state",
    "work_creativity_state",
    "team_trust_state",
]

# Canonical Big Five keys -> accepted column aliases (lowercased match).
TRAIT_ALIASES = {
    "extraversion":      ["extraversion", "bfi_extraversion", "e", "big5_e", "trait_extraversion"],
    "agreeableness":     ["agreeableness", "bfi_agreeableness", "a", "big5_a", "trait_agreeableness"],
    "conscientiousness": ["conscientiousness", "bfi_conscientiousness", "c", "big5_c", "trait_conscientiousness"],
    "neuroticism":       ["neuroticism", "bfi_neuroticism", "n", "big5_n", "trait_neuroticism",
                          "emotional_stability"],  # emotional_stability handled with a sign flip below
    "openness":          ["openness", "bfi_openness", "o", "big5_o", "trait_openness",
                          "openness_to_experience"],
}
# Aliases whose scale is REVERSED relative to the canonical trait (so we flip sign).
REVERSED_ALIASES = {"emotional_stability"}

# ══════════════════════════════════════════════════════════════════════════════
# NOMOLOGICAL BENCHMARK LIBRARY  (Big Five x 4 states)
# confidence: high = direct meta-analytic construct match; medium = close match /
# secondary source; low = inferred from nearest established path.
# DOI-verify before camera-ready (the anchor-verification pipeline).
# ══════════════════════════════════════════════════════════════════════════════
NOMO_BENCHMARKS = [
    # ── WORK ENGAGEMENT ──────────────────────────────────────────────────────
    {"trait": "extraversion", "state": "work_engagement_state", "expected_r": 0.27,
     "citation": "Young, Glerum, Wang & Joseph (2018) J. Organizational Behavior (meta)",
     "construct_match": "Extraversion -> Work engagement", "confidence": "high"},
    {"trait": "conscientiousness", "state": "work_engagement_state", "expected_r": 0.28,
     "citation": "Young et al. (2018) JOB meta",
     "construct_match": "Conscientiousness -> Work engagement", "confidence": "high"},
    {"trait": "neuroticism", "state": "work_engagement_state", "expected_r": -0.30,
     "citation": "Young et al. (2018) JOB meta",
     "construct_match": "Neuroticism -> Work engagement (negative)", "confidence": "high"},
    {"trait": "openness", "state": "work_engagement_state", "expected_r": 0.13,
     "citation": "Young et al. (2018) JOB meta",
     "construct_match": "Openness -> Work engagement", "confidence": "medium"},
    {"trait": "agreeableness", "state": "work_engagement_state", "expected_r": 0.17,
     "citation": "Young et al. (2018) JOB meta",
     "construct_match": "Agreeableness -> Work engagement", "confidence": "medium"},

    # ── PSYCHOLOGICAL SAFETY ─────────────────────────────────────────────────
    {"trait": "neuroticism", "state": "psych_safety_state", "expected_r": -0.25,
     "citation": "Frazier et al. (2017) Personnel Psychology (meta) + Edmondson (1999) ASQ",
     "construct_match": "Neuroticism -> Psychological safety (anxiety lowers felt safety)", "confidence": "medium"},
    {"trait": "extraversion", "state": "psych_safety_state", "expected_r": 0.18,
     "citation": "Frazier et al. (2017) Pers. Psych. (proactive personality path)",
     "construct_match": "Extraversion -> Psychological safety (voice/assertiveness)", "confidence": "low"},
    {"trait": "agreeableness", "state": "psych_safety_state", "expected_r": 0.20,
     "citation": "Frazier et al. (2017) Pers. Psych. (supportive context path)",
     "construct_match": "Agreeableness -> Psychological safety", "confidence": "low"},
    {"trait": "conscientiousness", "state": "psych_safety_state", "expected_r": 0.12,
     "citation": "Frazier et al. (2017) Pers. Psych. (role clarity path)",
     "construct_match": "Conscientiousness -> Psychological safety", "confidence": "low"},
    {"trait": "openness", "state": "psych_safety_state", "expected_r": 0.15,
     "citation": "Edmondson & Lei (2014) Annu. Rev. Org. Psych. (learning orientation)",
     "construct_match": "Openness -> Psychological safety (learning orientation)", "confidence": "low"},

    # ── WORK CREATIVITY ──────────────────────────────────────────────────────
    {"trait": "openness", "state": "work_creativity_state", "expected_r": 0.25,
     "citation": "Feist (1998) PSPR meta; Puryear, Kettler & Rinn (2017) J. Creative Behavior",
     "construct_match": "Openness -> Creativity", "confidence": "high"},
    {"trait": "extraversion", "state": "work_creativity_state", "expected_r": 0.12,
     "citation": "Puryear et al. (2017) J. Creative Behavior (meta)",
     "construct_match": "Extraversion -> Creativity", "confidence": "medium"},
    {"trait": "conscientiousness", "state": "work_creativity_state", "expected_r": -0.05,
     "citation": "Feist (1998) PSPR meta (weak negative for artistic/creative)",
     "construct_match": "Conscientiousness -> Creativity (weak negative)", "confidence": "medium"},
    {"trait": "neuroticism", "state": "work_creativity_state", "expected_r": -0.09,
     "citation": "Puryear et al. (2017) J. Creative Behavior (meta)",
     "construct_match": "Neuroticism -> Creativity (negative)", "confidence": "medium"},
    {"trait": "agreeableness", "state": "work_creativity_state", "expected_r": -0.05,
     "citation": "Feist (1998) PSPR meta (weak negative)",
     "construct_match": "Agreeableness -> Creativity (weak negative)", "confidence": "low"},

    # ── TEAM TRUST ───────────────────────────────────────────────────────────
    {"trait": "agreeableness", "state": "team_trust_state", "expected_r": 0.28,
     "citation": "Colquitt, Scott & LePine (2007) JAP meta (trust propensity)",
     "construct_match": "Agreeableness -> Trust (propensity)", "confidence": "high"},
    {"trait": "neuroticism", "state": "team_trust_state", "expected_r": -0.22,
     "citation": "Colquitt et al. (2007) JAP meta; Mooradian, Renzl & Matzler (2006)",
     "construct_match": "Neuroticism -> Trust (negative)", "confidence": "medium"},
    {"trait": "extraversion", "state": "team_trust_state", "expected_r": 0.15,
     "citation": "Mooradian et al. (2006) Management Learning",
     "construct_match": "Extraversion -> Trust (relational engagement)", "confidence": "low"},
    {"trait": "conscientiousness", "state": "team_trust_state", "expected_r": 0.14,
     "citation": "Colquitt et al. (2007) JAP meta (integrity/ability inference)",
     "construct_match": "Conscientiousness -> Trust", "confidence": "low"},
    {"trait": "openness", "state": "team_trust_state", "expected_r": 0.10,
     "citation": "Inferred from openness -> cooperative orientation",
     "construct_match": "Openness -> Trust", "confidence": "low"},
]


def detect_trait_columns(df: pd.DataFrame):
    """Map canonical trait -> (actual column, sign) using the alias table."""
    lower = {c.lower(): c for c in df.columns}
    found = {}
    for trait, aliases in TRAIT_ALIASES.items():
        for a in aliases:
            if a in lower:
                sign = -1.0 if a in REVERSED_ALIASES else 1.0
                found[trait] = (lower[a], sign)
                break
    return found


def test_nomological(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair observed trait->state convergence (between-person, Spearman)."""
    traits = detect_trait_columns(df)
    if not traits:
        raise ValueError("No Big Five trait columns detected. Expected one of the aliases in TRAIT_ALIASES.")
    state_cols = [c for c in STATE_COLS if c in df.columns]

    # Per-person state means + per-person (constant) trait.
    person = df.groupby("employee_id")[state_cols].mean()
    for trait, (col, sign) in traits.items():
        person[trait] = sign * df.groupby("employee_id")[col].first()

    rows = []
    for e in NOMO_BENCHMARKS:
        trait, st, exp = e["trait"], e["state"], e["expected_r"]
        if trait not in traits or st not in person.columns:
            continue
        d = person[[trait, st]].dropna()
        if len(d) < 10:
            continue
        rho = stats.spearmanr(d[trait], d[st]).correlation
        rows.append({
            "trait": trait, "state": st, "expected_r": exp,
            "observed_rho": round(float(rho), 4) if not np.isnan(rho) else np.nan,
            "dir_match": float(np.sign(rho) == np.sign(exp)) if not np.isnan(rho) else np.nan,
            "n_persons": int(len(d)),
            "confidence": e["confidence"], "citation": e["citation"],
            "construct_match": e["construct_match"],
        })
    return pd.DataFrame(rows)


def summarize(pairs: pd.DataFrame) -> dict:
    v = pairs.dropna(subset=["observed_rho"])
    if len(v) == 0:
        return {"n_pairs": 0}
    rho = (stats.spearmanr(v["observed_rho"], v["expected_r"]).correlation
           if len(v) >= 3 else np.nan)
    return {
        "n_pairs": int(len(v)),
        "dirmatch_pct": round(100 * v["dir_match"].mean(), 2),
        "convergence_rho": round(float(rho), 4) if not np.isnan(rho) else None,
        "dirmatch_high_conf_pct": round(
            100 * v[v["confidence"] == "high"]["dir_match"].mean(), 2)
            if (v["confidence"] == "high").any() else None,
    }


def make_demo(n_persons=120, n_days=6, rng_seed=11):
    """Stub corpus where states are generated FROM traits using the expected
    effects, so direction match should be near-perfect (a plumbing test, NOT a
    manuscript number)."""
    rng = np.random.default_rng(rng_seed)
    traits = {t: rng.normal(0, 1, n_persons) for t in TRAIT_ALIASES}
    scales = {"work_engagement_state": (1, 7), "psych_safety_state": (1, 4),
              "work_creativity_state": (1, 6), "team_trust_state": (1, 5)}
    # Build per-person latent state mean from the benchmark effects.
    latent = {st: np.zeros(n_persons) for st in scales}
    for e in NOMO_BENCHMARKS:
        latent[e["state"]] += e["expected_r"] * traits[e["trait"]]
    rows = []
    for pid in range(n_persons):
        base = {st: (lo + hi) / 2 + 1.2 * latent[st][pid] for st, (lo, hi) in scales.items()}
        for trait in TRAIT_ALIASES:
            pass
        for day in range(n_days):
            rec = {"employee_id": pid, "day": day}
            for t in TRAIT_ALIASES:
                rec[t] = float(traits[t][pid])
            for st, (lo, hi) in scales.items():
                rec[st] = float(np.clip(round(base[st] + rng.normal(0, 0.5), 2), lo, hi))
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Nomological trait->state convergence.")
    ap.add_argument("--corpus", default=None, help="CSV with *_state cols + Big Five trait cols.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo or not args.corpus:
        df = make_demo()
        label = args.label + " (DEMO STUB, not a manuscript number)"
    else:
        df = pd.read_csv(args.corpus)
        label = args.label

    pairs = test_nomological(df)
    pairs.to_csv(out / "nomological_pairs.csv", index=False)
    summ = summarize(pairs)
    pd.DataFrame([summ]).to_csv(out / "nomological_summary.csv", index=False)

    print(f"Nomological [{label}]")
    print(f"  pairs={summ.get('n_pairs')}  dir-match={summ.get('dirmatch_pct')}%  "
          f"convergence rho={summ.get('convergence_rho')}  "
          f"high-conf dir={summ.get('dirmatch_high_conf_pct')}%")

    with open(out / "RESULTS_nomological.md", "w") as fh:
        fh.write(f"# Nomological trait->state convergence — {label}\n\n")
        fh.write("Generated by `scripts/validation/nomological.py`.\n\n")
        fh.write("- Anchors: `NOMO_BENCHMARKS` (Big Five x 4 states) defined in this module.\n")
        fh.write("- Convergence = direction-match %% and signed Spearman rho of observed "
                 "between-person trait->state rho vs published expected_r.\n\n")
        fh.write("| metric | value |\n|---|---|\n")
        for k, v in summ.items():
            fh.write(f"| {k} | {v} |\n")
    print(f"Wrote {out}/nomological_pairs.csv, nomological_summary.csv, RESULTS_nomological.md")


if __name__ == "__main__":
    main()
