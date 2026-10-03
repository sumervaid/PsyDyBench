#!/usr/bin/env python3
"""
Re-anchor the convergence leaderboard on FULL-TEXT-VERIFIED OBSERVED r (vs the original claimed
targets, which were mostly corrected rho). Quantifies how much the corrected->observed correction
moves convergence.

For each verified pair (status==verified, r_observed not null in data/seeds/fulltext_extractions.json):
  override that META_BENCHMARKS entry's meta_r with r_observed.
Then recompute, per model, the pooled/within direction-match% and signed Spearman rho on BOTH the
original and the re-anchored target set, and report the delta. Also a per-pair table: simulated r
(pooled) vs claimed target vs verified observed r, for the verified pairs.

Inputs : results/generation/<run>/<MODEL>/states.csv ; data/seeds/personas_full.csv ;
         scripts/validation/meta_analysis_validation.py ; data/seeds/fulltext_extractions.json
Outputs: results/leaderboard/reanchored_convergence.csv ; results/leaderboard/REANCHOR_RESULTS.md
Usage  : python scripts/analysis/reanchor_convergence.py results/generation/full_v2
"""
import sys, glob, json, csv, copy
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/validation"))
from meta_analysis_validation import META_BENCHMARKS, compute_pooled_r, compute_within_r  # noqa

STATE_MAP = {"work_engagement": "work_engagement_state", "psych_safety": "psych_safety_state",
             "work_creativity": "work_creativity_state", "team_trust": "team_trust_state"}

def verified_overrides():
    d = json.load(open(REPO / "data/seeds/fulltext_extractions.json"))
    ov = {}
    for p in d["papers"].values():
        for r in p["pairs"]:
            if r["status"] == "verified" and r["r_observed"] is not None:
                ov[(r["our_indicator"], r["our_state"] + "_state")] = (r["r_observed"], r["claimed"])
    return ov

def summarise(pairs):
    v = [(o, t) for o, t in pairs if o == o]
    if len(v) < 3: return None, None, len(v)
    dirm = 100 * np.mean([np.sign(o) == np.sign(t) for o, t in v])
    rho = stats.spearmanr([o for o, t in v], [t for o, t in v]).correlation
    return round(float(dirm), 1), round(float(rho), 3), len(v)

def convergence(m, bench):
    pooled, within = [], []
    for e in bench:
        i, s, mr = e["indicator"], e["state"], e["meta_r"]
        if i in m.columns and s in m.columns:
            pooled.append((compute_pooled_r(m, i, s), mr))
            within.append((compute_within_r(m, i, s), mr))
    pd_, pr_, n = summarise(pooled); wd_, wr_, _ = summarise(within)
    return pd_, pr_, wd_, wr_, n

def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/generation/full_v2")
    ov = verified_overrides()
    bench_re = copy.deepcopy(META_BENCHMARKS)
    nov = 0
    for e in bench_re:
        k = (e["indicator"], e["state"])
        if k in ov:
            e["meta_r"] = ov[k][0]; nov += 1
    ind = pd.read_csv(REPO / "data/seeds/personas_full.csv")
    icols = [c for c in ind.columns if c.split("_")[0] in
             ("behavioral", "physical", "social", "psychological", "digital", "life")]
    ind = ind[["employee_id", "day"] + icols]

    rows, perpair_done = [], False
    md_pairs = ["## Per-pair: simulated r vs claimed target vs verified observed r",
                "(pooled/between-person simulated r, averaged across models; the 21 verified pairs)", "",
                "| pair | claimed target | verified obs r | mean simulated r |", "|---|---|---|---|"]
    sim_accum = {k: [] for k in ov}

    for sc in sorted(glob.glob(str(base / "*/states.csv"))):
        model = Path(sc).parent.name
        st = pd.read_csv(sc).rename(columns={"day_index": "day", **STATE_MAP})
        keep = ["employee_id", "day"] + [c for c in STATE_MAP.values() if c in st.columns]
        m = ind.merge(st[keep], on=["employee_id", "day"], how="inner")
        op = convergence(m, META_BENCHMARKS); rp = convergence(m, bench_re)
        rows.append({"model": model,
                     "orig_pooled_dir%": op[0], "reanch_pooled_dir%": rp[0],
                     "orig_pooled_rho": op[1], "reanch_pooled_rho": rp[1],
                     "orig_within_dir%": op[2], "reanch_within_dir%": rp[2], "n_pairs": op[4]})
        for (i, s), (obs, claimed) in ov.items():
            if i in m.columns and s in m.columns:
                sim_accum[(i, s)].append(compute_pooled_r(m, i, s))

    for (i, s), (obs, claimed) in ov.items():
        sims = [x for x in sim_accum[(i, s)] if x == x]
        msim = round(float(np.mean(sims)), 3) if sims else None
        md_pairs.append(f"| {i} -> {s} | {claimed} | {obs} | {msim} |")

    rows.sort(key=lambda r: -(r["reanch_pooled_dir%"] or 0))
    out = REPO / "results/leaderboard/reanchored_convergence.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # deltas
    dd = [(r["reanch_pooled_dir%"] - r["orig_pooled_dir%"]) for r in rows
          if r["reanch_pooled_dir%"] is not None and r["orig_pooled_dir%"] is not None]
    md = [f"# Re-anchored convergence: observed r vs claimed targets",
          f"\n{nov} of {len(ov)} verified pairs overrode their META_BENCHMARKS target with full-text "
          f"OBSERVED r (the rest of the ~118-pair library unchanged). Models scored: {len(rows)} "
          f"(preliminary — full 48-model run still finishing).",
          f"\nMean change in pooled direction-match%: {np.mean(dd):+.2f} pts (per model).", "",
          "## Per-model: original (claimed targets) vs re-anchored (observed r)", "",
          "| model | orig dir% | reanch dir% | orig rho | reanch rho |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['orig_pooled_dir%']} | {r['reanch_pooled_dir%']} | "
                  f"{r['orig_pooled_rho']} | {r['reanch_pooled_rho']} |")
    md += ["", *md_pairs, "",
           "**Reading:** if direction-match barely moves, the convergence finding is robust to the "
           "corrected-vs-observed target choice (the SIGNS agree even where magnitudes differ). "
           "Magnitude differences show up in the per-pair table (claimed=corrected rho tends to exceed "
           "observed r)."]
    (REPO / "results/leaderboard/REANCHOR_RESULTS.md").write_text("\n".join(md) + "\n")
    print(f"overrode {nov}/{len(ov)} verified pairs; scored {len(rows)} models; "
          f"mean pooled dir% change {np.mean(dd):+.2f}")
    print("wrote", out)

if __name__ == "__main__":
    main()
