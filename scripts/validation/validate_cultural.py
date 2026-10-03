#!/usr/bin/env python3
"""
Cross-cultural validation: (1) measurement invariance and (2) cross-cultural convergence.

Reads results/generation/cultural/<MODEL>/states.csv (culture encoded in employee_id prefix)
+ data/seeds/cultural_norms.json. Writes results/leaderboard/cultural_validation.csv and
results/leaderboard/CULTURAL_RESULTS.md.

(1) Measurement invariance (metric-invariance proxy): for each culture compute the 4x4 cross-state
    correlation matrix; report the mean |Δr| of each culture's off-diagonals from the grand-mean
    matrix. Low = the psychometric structure is stable across cultures (configural/metric-invariant).
(2) Cross-cultural convergence: across the N cultures, correlate the culture's Hofstede dimension
    (PDI or IDV) with its mean state, and check the sign against the published anchor's expected sign
    (cross_cultural_anchors). Report direction-match % + per-anchor r.

Usage: python3 validate_cultural.py results/generation/cultural
"""
import sys, glob, json, csv
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CN = json.load(open(REPO / "data/seeds/cultural_norms.json"))
STATES = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]

def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "results/generation/cultural")
    cult = CN["cultures"]; anchors = CN["cross_cultural_anchors"]
    iu = np.triu_indices(len(STATES), 1)
    rows, md = [], ["# Cross-cultural validation results", ""]

    for sc in sorted(glob.glob(str(base / "*/states.csv"))):
        model = Path(sc).parent.name
        df = pd.read_csv(sc)
        if not set(STATES).issubset(df.columns):
            md.append(f"## {model}\n- SKIP: states.csv missing columns\n"); continue
        df["culture"] = df["employee_id"].astype(str).str.split("_").str[0]
        cmean = df.groupby("culture")[STATES].mean()

        # (1) measurement invariance
        mats = {c: g[STATES].corr().values for c, g in df.groupby("culture") if len(g) > 10}
        invar = float("nan")
        if len(mats) >= 2:
            grand = np.mean(list(mats.values()), axis=0)
            invar = float(np.mean([np.mean(np.abs(m[iu] - grand[iu])) for m in mats.values()]))

        # (2) cross-cultural convergence vs anchors
        conv = []
        for an in anchors:
            dim, st, exp = an["dimension"], an["state"], an["expected_sign"]
            if st not in cmean.columns:
                continue
            x = np.array([cult[c][dim] for c in cmean.index], float)
            y = cmean[st].values
            if np.std(x) > 0 and np.std(y) > 0:
                r = float(np.corrcoef(x, y)[0, 1])
                conv.append((an, r, int(np.sign(r) == exp)))
        dirmatch = 100 * np.mean([m for *_, m in conv]) if conv else float("nan")

        rows.append({"model": model, "n_cultures": len(mats),
                     "invariance_mean_abs_dr": round(invar, 3) if invar == invar else "",
                     "xcult_conv_dirmatch_pct": round(dirmatch, 1) if dirmatch == dirmatch else "",
                     "n_anchors": len(conv)})
        md.append(f"## {model}")
        md.append(f"- **Measurement invariance**: mean |Δr| of per-culture cross-state correlations vs "
                  f"grand = {invar:.3f} (lower = more invariant; ~<0.10 = strong metric invariance)")
        md.append(f"- **Cross-cultural convergence**: {dirmatch:.0f}% of {len(conv)} Hofstede anchors "
                  f"match the published direction")
        for an, r, m in conv:
            md.append(f"  - {an['dimension'].upper()} → {an['state']} (expected {an['expected_sign']:+d}, "
                      f"{an['confidence']}): r={r:+.2f} {'✓' if m else '✗'}")
        md.append("\n  per-culture mean states:")
        md.append("  | culture | " + " | ".join(STATES) + " |")
        md.append("  |" + "---|" * (len(STATES) + 1))
        for c in cmean.index:
            md.append("  | " + c + " | " + " | ".join(f"{cmean.loc[c, s]:.2f}" for s in STATES) + " |")
        md.append("")

    if not rows:
        print("no model states found in", base); return
    out = REPO / "results/leaderboard/cultural_validation.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (REPO / "results/leaderboard/CULTURAL_RESULTS.md").write_text("\n".join(md) + "\n")
    print("wrote", out)
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
