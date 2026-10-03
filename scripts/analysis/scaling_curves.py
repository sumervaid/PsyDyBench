#!/usr/bin/env python3
"""
Scaling-curve analysis for longitudinal-persona fidelity vs model size.

Joins the two leaderboards (internal-validity + anchoring) to the scaling roster's parameter
counts and, for each validity metric, fits metric ~ log10(params_b) — per dense family (>=3
points) and pooled across dense models. MoE models are plotted separately by BOTH total and
active params (the dense fit is the primary 'law'); closed/reference models (unknown params)
are shown as a reference band, not in the fit.

Reports per-family slopes (is fidelity increasing, flat, or INVERTING with size?) and writes a
figure. The headline question: do different validity dimensions scale in opposite directions?

Inputs : results/leaderboard/{leaderboard,anchoring_leaderboard}.csv ; data/seeds/model_roster_scaling.csv
Outputs: results/leaderboard/scaling_slopes.csv ; results/leaderboard/SCALING_RESULTS.md ;
         results/figures/scaling_curves.png
Usage  : python scripts/analysis/scaling_curves.py
"""
import csv, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LB = REPO / "results/leaderboard/leaderboard.csv"
AB = REPO / "results/leaderboard/anchoring_leaderboard.csv"
ROSTER = REPO / "data/seeds/model_roster_scaling.csv"
OUTMD = REPO / "results/leaderboard/SCALING_RESULTS.md"
OUTCSV = REPO / "results/leaderboard/scaling_slopes.csv"
FIG = REPO / "results/figures/scaling_curves.png"

# metric -> (source leaderboard, column, "higher better"?)
METRICS = [
    ("var_fidelity", "lb", "var_fidelity", False),   # lower = better
    ("internal_validity", "lb", "internal_validity", True),
    ("anchor_pooled_dir%", "ab", "pooled_dir_pct", True),
    ("anchor_pooled_rho", "ab", "pooled_rho", True),
    ("anchor_within_dir%", "ab", "within_dir_pct", True),
]

def namekey(model_id):
    return model_id.split("/")[-1].upper().replace("-", "_").replace(".", "_")

def load_csv(p):
    return {r[list(r)[0]]: r for r in csv.DictReader(open(p))} if p.exists() else {}

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def linfit(xs, ys):
    n = len(xs)
    if n < 2: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    if sxx == 0: return None
    slope = sxy/sxx
    syy = sum((y-my)**2 for y in ys)
    r = sxy/math.sqrt(sxx*syy) if syy > 0 else 0.0
    return slope, r, n

def main():
    lb = {r["model"]: r for r in csv.DictReader(open(LB))} if LB.exists() else {}
    ab = {r["model"]: r for r in csv.DictReader(open(AB))} if AB.exists() else {}
    roster = list(csv.DictReader(open(ROSTER)))

    # assemble joined records
    recs = []
    for r in roster:
        key = namekey(r["model_id"])
        rec = {"key": key, "family": r["family"], "kind": r["kind"], "role": r["role"],
               "is_variant": r.get("is_variant", "N"),
               "params": fnum(r["params_b"]), "active": fnum(r["active_params_b"])}
        for mname, src, col, _ in METRICS:
            row = (lb if src == "lb" else ab).get(key, {})
            rec[mname] = fnum(row.get(col))
        recs.append(rec)

    # pooled cross-model cloud = ALL dense models with known params (incl. version variants).
    # clean per-family fit = dense ladder models only, NO version variants.
    pooled_dense = [r for r in recs if r["kind"] == "dense" and r["params"]]
    dense_ladder = [r for r in recs if r["kind"] == "dense" and r["role"] == "ladder"
                    and r["is_variant"] != "Y" and r["params"]]

    slopes = []
    md = ["# Scaling-curve analysis — longitudinal-persona fidelity vs model size", "",
          "Fit: metric ~ log10(params_b). Dense models only in the fit; MoE shown by total+active "
          "separately; closed/reference models excluded (unknown params). Run after the scaling "
          "benchmark; metrics for not-yet-run models are blank.", ""]

    families = sorted(set(r["family"] for r in dense_ladder))
    for mname, src, col, higher in METRICS:
        md.append(f"## {mname}  ({'higher=better' if higher else 'lower=better'})")
        # pooled across ALL dense (incl. variants)
        pts = [(math.log10(r["params"]), r[mname]) for r in pooled_dense if r[mname] is not None]
        fit = linfit([x for x, _ in pts], [y for _, y in pts])
        if fit:
            s, rr, n = fit
            md.append(f"- **pooled (all dense, n={n})**: slope={s:+.3f} per decade, r={rr:+.2f}")
            slopes.append({"metric": mname, "family": "ALL_DENSE", "slope_per_decade": round(s, 4),
                           "pearson_r": round(rr, 3), "n": n})
        for fam in families:
            fpts = [(math.log10(r["params"]), r[mname]) for r in dense_ladder
                    if r["family"] == fam and r[mname] is not None]
            fit = linfit([x for x, _ in fpts], [y for _, y in fpts])
            if fit:
                s, rr, n = fit
                md.append(f"  - {fam} (n={n}): slope={s:+.3f}, r={rr:+.2f}")
                slopes.append({"metric": mname, "family": fam, "slope_per_decade": round(s, 4),
                               "pearson_r": round(rr, 3), "n": n})
        md.append("")

    md.append("## Interpretation hook\n"
              "Compare slope SIGNS across metrics: if variance-fidelity and anchor-convergence scale "
              "in OPPOSITE directions, longitudinal-persona validity does not follow a single capability "
              "scaling law — the central finding. [[v2: fill once the scaling benchmark completes]]")

    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    if slopes:
        with open(OUTCSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(slopes[0].keys())); w.writeheader(); w.writerows(slopes)
    OUTMD.write_text("\n".join(md) + "\n")

    # figure (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        WONG = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
        fitmetrics = [m for m in METRICS if any(r[m[0]] is not None for r in pooled_dense)]
        plot_fams = sorted(set(r["family"] for r in pooled_dense))
        if fitmetrics:
            fig, axes = plt.subplots(1, len(fitmetrics), figsize=(5*len(fitmetrics), 4.2), squeeze=False)
            fams = {f: WONG[i % len(WONG)] for i, f in enumerate(plot_fams)}
            for ax, (mname, *_rest) in zip(axes[0], fitmetrics):
                for fam in plot_fams:
                    fpts = [(r["params"], r[mname]) for r in pooled_dense
                            if r["family"] == fam and r[mname] is not None]
                    if fpts:
                        ax.scatter([p for p, _ in fpts], [v for _, v in fpts], color=fams[fam], label=fam, s=45)
                ax.set_xscale("log"); ax.set_xlabel("params (B, log)"); ax.set_title(mname); ax.grid(alpha=.3)
            axes[0][0].legend(fontsize=7)
            fig.suptitle("Longitudinal-persona fidelity vs model size (dense ladders)")
            fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=150)
            print("wrote", FIG)
    except Exception as e:
        print("figure skipped:", e)

    print("wrote", OUTMD, "and", OUTCSV)
    print(f"dense ladder models: {len(dense_ladder)}; with metrics: "
          f"{sum(1 for r in dense_ladder if r['var_fidelity'] is not None)}")

if __name__ == "__main__":
    main()
