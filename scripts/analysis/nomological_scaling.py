#!/usr/bin/env python3
"""
Per-model NOMOLOGICAL external-validity (the clean BETWEEN-person check) + KNOWN-GROUPS validity,
and how both scale with model size.

WHY this is the strong external check: traits (Big Five) are seeded with real between-person variance
(MVN from published norms), so trait->state effects are recoverable at the between-person level against
published targets — unlike the daily indicators (IID-seeded, no between-person signal). This is
estimand-clean construct validity.

Three results per model:
  (1) Nomological direction agreement: for each of the 26 (trait, state) NOMO_BENCHMARKS pairs, correlate
      the persona's trait with their person-mean state (between-person), score sign-match vs the published
      expected_r + signed Spearman rho across pairs.
  (2) Known-groups: high vs low Neuroticism personas should differ in state MEANS (lower safety/engagement)
      and day-to-day VOLATILITY (higher within-person SD). Report deltas.
  (3) Scaling: correlate model params_b (dense models) with nomological direction% -> does external
      validity scale with size?

Inputs : results/generation/<run>/<MODEL>/states.csv ; data/seeds/personas_full.csv ;
         scripts/validation/nomological.py (NOMO_BENCHMARKS) ; data/seeds/model_roster_scaling.csv
Outputs: results/leaderboard/nomological_leaderboard.csv ; results/leaderboard/NOMOLOGICAL_RESULTS.md
Usage  : python scripts/analysis/nomological_scaling.py results/generation/full_v2
"""
import sys, glob, csv, math
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/validation"))
from nomological import NOMO_BENCHMARKS  # noqa

BIG5 = ["extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness"]
STATES = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]

def trait_series(persons, trait):
    """Return per-person trait values; map neuroticism = reverse of emotional_stability."""
    if trait == "neuroticism":
        return 8.0 - persons["emotional_stability"] if "emotional_stability" in persons else None
    return persons[trait] if trait in persons else None

def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/generation/full_v2")
    pf = pd.read_csv(REPO / "data/seeds/personas_full.csv")
    traits = pf.groupby("employee_id")[BIG5].first()  # stable per persona
    roster = {r["model_id"].split("/")[-1].upper().replace("-", "_").replace(".", "_"): r
              for r in csv.DictReader(open(REPO / "data/seeds/model_roster_scaling.csv"))}

    rows, kg_rows = [], []
    for sc in sorted(glob.glob(str(base / "*/states.csv"))):
        model = Path(sc).parent.name
        st = pd.read_csv(sc).rename(columns={"day_index": "day"})
        scol = [c for c in STATES if c in st.columns]
        if not scol: continue
        pmean = st.groupby("employee_id")[scol].mean()           # person-mean state (between)
        pvol = st.groupby("employee_id")[scol].std()             # within-person volatility
        m = traits.join(pmean, how="inner")
        if len(m) < 30: continue

        # (1) nomological direction agreement
        obs, exp = [], []
        for e in NOMO_BENCHMARKS:
            tr, s_full, er = e["trait"], e["state"], e["expected_r"]
            s = s_full.replace("_state", "")
            tv = trait_series(m, tr)
            if tv is None or s not in m.columns: continue
            if tv.std() == 0 or m[s].std() == 0: continue
            r = tv.corr(m[s])
            obs.append(r); exp.append(er)
        if len(obs) < 3: continue
        dirm = 100 * np.mean([np.sign(o) == np.sign(t) for o, t in zip(obs, exp)])
        rho = stats.spearmanr(obs, exp).correlation
        params = roster.get(model, {}).get("params_b", "")
        rows.append({"model": model, "nomo_dir_pct": round(float(dirm), 1),
                     "nomo_rho": round(float(rho), 3), "n_pairs": len(obs),
                     "params_b": params, "kind": roster.get(model, {}).get("kind", "")})

        # (2) known-groups: high vs low neuroticism
        neu = trait_series(m.join(pf.groupby("employee_id")["emotional_stability"].first(), rsuffix="_x")
                           if "emotional_stability" not in m else m, "neuroticism")
        if neu is not None and "work_engagement" in pmean.columns:
            md = neu.median(); hi, lo = neu > md, neu <= md
            def delta(df, col):
                return round(float(df.loc[hi.index[hi], col].mean() - df.loc[lo.index[lo], col].mean()), 3) if col in df else None
            kg_rows.append({"model": model,
                "hiN-loN engagement(mean)": delta(pmean, "work_engagement"),
                "hiN-loN psych_safety(mean)": delta(pmean, "psych_safety"),
                "hiN-loN engagement(volatility)": delta(pvol, "work_engagement")})

    rows.sort(key=lambda r: -(r["nomo_dir_pct"] or 0))
    out = REPO / "results/leaderboard/nomological_leaderboard.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # (3) scaling tie-in: params vs nomo_dir% (dense models with params)
    dn = [(math.log10(float(r["params_b"])), r["nomo_dir_pct"]) for r in rows
          if r["kind"] == "dense" and r["params_b"] not in ("", None)]
    slope = rr = None
    if len(dn) >= 3:
        xs = [x for x, _ in dn]; ys = [y for _, y in dn]
        slope = round(float(np.polyfit(xs, ys, 1)[0]), 2)
        rr = round(float(np.corrcoef(xs, ys)[0, 1]), 3)

    means = [r["nomo_dir_pct"] for r in rows if r["nomo_dir_pct"] is not None]
    md = ["# Nomological external validity (trait->state, BETWEEN-person) + known-groups", "",
          "**The clean external check.** Big Five traits are seeded with real between-person variance, so "
          "trait->state effects are recoverable against published targets (unlike IID daily indicators). "
          "This is estimand-matched construct validity.", "",
          f"Models scored: {len(rows)}.  Mean nomological direction-match: **{np.mean(means):.1f}%** "
          f"(chance 50%), range [{min(means):.0f}, {max(means):.0f}].", "",
          "## Leaderboard (trait->state direction agreement)", "",
          "| model | nomo dir% | nomo rho | n pairs | params(B) |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['nomo_dir_pct']} | {r['nomo_rho']} | {r['n_pairs']} | {r['params_b']} |")
    md += ["", "## Scaling tie-in",
           f"Dense-model fit: nomological direction% ~ log10(params): slope={slope} pts/decade, r={rr} "
           f"(n={len(dn)}). [[PENDING: refit on full 48-model roster.]]",
           "This is the FOURTH validity dimension for the scaling analysis (alongside variance fidelity, "
           "internal validity, indicator convergence) — does EXTERNAL nomological validity track size?", "",
           "## Known-groups validity (high vs low Neuroticism)",
           "Expectation: higher Neuroticism -> lower engagement/safety MEAN and higher day-to-day VOLATILITY.", "",
           "| model | Δengagement mean | Δpsych_safety mean | Δengagement volatility |", "|---|---|---|---|"]
    for k in kg_rows:
        md.append(f"| {k['model']} | {k['hiN-loN engagement(mean)']} | {k['hiN-loN psych_safety(mean)']} "
                  f"| {k['hiN-loN engagement(volatility)']} |")
    (REPO / "results/leaderboard/NOMOLOGICAL_RESULTS.md").write_text("\n".join(md) + "\n")
    print(f"scored {len(rows)} models; mean nomo dir% {np.mean(means):.1f}; scaling slope {slope} r {rr}")
    print("wrote", out)

if __name__ == "__main__":
    main()
