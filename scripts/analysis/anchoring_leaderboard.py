#!/usr/bin/env python3
"""
Per-model META-ANALYTIC CONVERGENCE leaderboard (Framework 2).

For each benchmarked model, join the seeded 55 indicators (personas_full.csv) to that
model's GENERATED states, compute the 118 indicator->state correlations, and score
convergence vs the published META_BENCHMARKS effects: direction-match % + signed Spearman
rho, BOTH pooled/between-person and within-person.

This is the second leaderboard (which models reproduce the published effect structure),
to compare against the internal-validity leaderboard.

Usage: python scripts/analysis/anchoring_leaderboard.py results/generation/full
Output: results/leaderboard/anchoring_leaderboard.csv
"""
import sys, glob, csv
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/validation"))
from meta_analysis_validation import META_BENCHMARKS, compute_pooled_r, compute_within_r, compute_between_r  # noqa

STATE_MAP = {"work_engagement": "work_engagement_state", "psych_safety": "psych_safety_state",
             "work_creativity": "work_creativity_state", "team_trust": "team_trust_state"}

def summarise(pairs):
    v = [(o, t) for o, t in pairs if o == o]
    if len(v) < 3:
        return None, None, len(v)
    dirm = 100 * np.mean([np.sign(o) == np.sign(t) for o, t in v])
    rho = stats.spearmanr([o for o, t in v], [t for o, t in v]).correlation
    return round(float(dirm), 1), round(float(rho), 3), len(v)

def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/generation/full")
    ind = pd.read_csv(REPO / "data/seeds/personas_full.csv")
    icols = [c for c in ind.columns if c.split("_")[0] in
             ("behavioral", "physical", "social", "psychological", "digital", "life")]
    ind = ind[["employee_id", "day"] + icols]

    rows = []
    for sc in sorted(glob.glob(str(base / "*/states.csv"))):
        model = Path(sc).parent.name
        st = pd.read_csv(sc).rename(columns={"day_index": "day", **STATE_MAP})
        st = st[["employee_id", "day"] + list(STATE_MAP.values())]
        m = ind.merge(st, on=["employee_id", "day"], how="inner")
        pooled, within, between = [], [], []
        for e in META_BENCHMARKS:
            i, s, mr = e["indicator"], e["state"], e["meta_r"]
            if i in m.columns and s in m.columns:
                pooled.append((compute_pooled_r(m, i, s), mr))
                within.append((compute_within_r(m, i, s), mr))
                between.append((compute_between_r(m, i, s), mr))  # meta-matched estimand
        pd_, pr_, n = summarise(pooled)
        wd_, wr_, _ = summarise(within)
        bd_, br_, _ = summarise(between)
        rows.append({"model": model,
                     "between_dir_pct": bd_, "between_rho": br_,   # MATCHES the meta-analytic estimand
                     "pooled_dir_pct": pd_, "pooled_rho": pr_,
                     "within_dir_pct": wd_, "within_rho": wr_, "n_pairs": n})
    rows.sort(key=lambda r: -(r["between_dir_pct"] or 0))
    out = REPO / "results/leaderboard/anchoring_leaderboard.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'model':30s} {'BETWEEN_dir%':>12s} {'BETWEEN_rho':>11s} {'pooled_dir%':>11s} {'within_dir%':>11s}")
    for r in rows:
        print(f"{r['model']:30s} {str(r['between_dir_pct']):>12s} {str(r['between_rho']):>11s} "
              f"{str(r['pooled_dir_pct']):>11s} {str(r['within_dir_pct']):>11s}")
    print(f"\nn_pairs={rows[0]['n_pairs']}. BETWEEN matches the published meta_r estimand (across-person).")
    print("If between_dir% ~ chance/between_rho ~ 0, the IID-seeded indicators carry no between-person")
    print("trait-linked signal -> meta convergence is not recoverable at the between level by design.")

if __name__ == "__main__":
    main()
