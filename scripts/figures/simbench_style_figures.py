#!/usr/bin/env python3
"""
Publication figures for the paper, styled EXACTLY on SimBench (Hu et al., ICLR 2026):

  Fig 2 (scaling)   -> fig_scaling.pdf : parameter count (log x) vs validity (y), one line per
                       model family, markers per model. Two panels: nomological direction% and
                       within-person variability calibration. Mirrors SimBench Fig 2.
  Fig 3 (by criterion) -> fig_bycriterion.pdf : grouped bars, internal-validity criteria (x) x top-5 models,
                       per-criterion score (y). Mirrors SimBench Fig 3 (fidelity by dataset, top-5).
  Fig 4 (tradeoff)  -> fig_tradeoff.pdf : scatter with fitted line + r annotation. x = within-person
                       variability calibration (internal), y = nomological direction% (external),
                       one point per model. Mirrors SimBench Fig 4 (tradeoff scatter + regression).

Clean style: Helvetica-like sans, no top/right spines, light grid, colorblind-safe family palette,
sorted/annotated. Outputs PDF (vector) + PNG to results/figures/.
Usage: python scripts/figures/simbench_style_figures.py
"""
import csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parents[2]
LB = REPO / "results/leaderboard"
OUT = REPO / "results/figures"; OUT.mkdir(parents=True, exist_ok=True)

# --- SimBench-like aesthetic ---
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False, "legend.fontsize": 7.5,
})
# colorblind-safe palette per family (Wong)
FAMCOL = {"Llama-3.x": "#0072B2", "Gemma-3": "#D55E00", "Qwen2.5": "#009E73",
          "Qwen3": "#CC79A7", "Gemma-2": "#E69F00", "Mistral": "#56B4E9", "Phi": "#000000"}

def load(name):
    return list(csv.DictReader(open(LB / name)))

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def roster():
    r = {}
    for row in csv.DictReader(open(REPO / "data/seeds/model_roster_scaling_live.csv")):
        key = row["model_id"].split("/")[-1].upper().replace("-", "_").replace(".", "_")
        r[key] = row
    return r

def clean_label(m):
    return (m.replace("_INSTRUCT", "").replace("_IT", "").replace("_", "-").title()
             .replace("Gpt", "GPT").replace("Gemma", "Gemma").replace("Qwen", "Qwen"))

# ---------------------------------------------------------------- Fig 2: scaling
def fig_scaling():
    nomo = {r["model"]: r for r in load("nomological_leaderboard.csv")}
    lb = {r["model"]: r for r in load("leaderboard.csv")}
    rost = roster()
    pts = []
    for m, r in nomo.items():
        p = fnum(r.get("params_b")); fam = r.get("kind") and rost.get(m, {}).get("family")
        if p is None or rost.get(m, {}).get("kind") != "dense": continue
        fam = rost[m]["family"]
        nd = fnum(r.get("nomo_dir_pct")); vf = fnum(lb.get(m, {}).get("var_fidelity"))
        if fam in FAMCOL and nd is not None:
            pts.append((fam, p, nd, vf))
    fams = sorted({f for f, *_ in pts})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for j, (metric, ylab, idx, inv) in enumerate([
            ("nomo", "Nomological direction agreement (%)", 2, False),
            ("vf", "Within-person variance error", 3, True)]):
        for fam in fams:
            fp = sorted([(p, (nd if idx == 2 else vf)) for (f, p, nd, vf) in pts
                         if f == fam and (nd if idx == 2 else vf) is not None], key=lambda t: t[0])
            if len(fp) >= 2:
                xs, ys = zip(*fp)
                ax[j].plot(xs, ys, "-o", color=FAMCOL[fam], ms=4, lw=1.4, label=fam)
            elif fp:
                ax[j].plot(*zip(*fp), "o", color=FAMCOL[fam], ms=4, label=fam)
        ax[j].set_xscale("log"); ax[j].set_xlabel("Parameters (B, log scale)"); ax[j].set_ylabel(ylab)
        if inv: ax[j].invert_yaxis()
    ax[0].axhline(50, ls="--", lw=0.8, color="grey"); ax[0].text(ax[0].get_xlim()[0], 51, "chance", fontsize=6.5, color="grey")
    ax[0].legend(loc="lower right", ncol=2)
    fig.suptitle("Validity vs. model scale, by family", y=1.02, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_scaling.{ext}")
    plt.close(fig); print("wrote fig_scaling")

# --------------------------------------------- Fig 3: per-state variability calibration, top-5
def fig_bycriterion():
    """Per-state within-person variability calibration |ln(synth_wsd/target_wsd)| (lower=better,
    a single continuous metric on a common scale) for the top-5 models by overall calibration.
    Mirrors SimBench Fig 3 (one continuous fidelity score across categories x top-5 models)."""
    import json
    states = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]
    slab = {"work_engagement": "Engagement", "psych_safety": "Psych. safety",
            "work_creativity": "Creativity", "team_trust": "Team trust"}
    # rank models by overall var_fidelity (lower=better), drop degenerate (>1.0)
    lb = [r for r in load("leaderboard.csv") if fnum(r.get("var_fidelity")) is not None
          and fnum(r["var_fidelity"]) <= 1.0]
    lb.sort(key=lambda r: fnum(r["var_fidelity"]))
    cal = {}  # model -> {state: |ln ratio|}
    order = []
    for r in lb:
        m = r["model"]; fp = LB / f"{m}_internal_validity.json"
        if not fp.exists(): continue
        d = json.loads(fp.read_text())
        per = {}
        for lv in d["levels"]:
            if lv["level"] == 2:
                for s, mm in lv["metrics"].items():
                    sw, tw = mm.get("synth_within_sd"), mm.get("target_within_sd")
                    if sw and tw and sw > 0 and tw > 0:
                        per[s] = abs(math.log(sw / tw))
        if len(per) == len(states):
            cal[m] = per; order.append(m)
        if len(order) == 5: break
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(states)); w = 0.8 / max(len(order), 1)
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for i, m in enumerate(order):
        vals = [cal[m][s] for s in states]
        ax.bar(x + i*w - 0.4 + w/2, vals, w, color=cols[i % 5], label=clean_label(m))
    ax.axhline(math.log(1.5), ls="--", lw=0.8, color="grey")
    ax.text(-0.45, math.log(1.5)+0.01, "equiv. margin (1.5$\\times$)", fontsize=6.5, color="grey")
    ax.set_xticks(x); ax.set_xticklabels([slab[s] for s in states])
    ax.set_ylabel("Within-person variance error\n$|\\ln(\\mathrm{SD_{synth}}/\\mathrm{SD_{target}})|$ (lower better)")
    ax.set_title("Within-person variance error by state (top-5 models)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_bycriterion.{ext}")
    plt.close(fig); print("wrote fig_bycriterion (per-state calibration)")

# ---------------------------------------------------------------- Fig 4: tradeoff + fit + r
def fig_tradeoff():
    nomo = {r["model"]: r for r in load("nomological_leaderboard.csv")}
    lb = {r["model"]: r for r in load("leaderboard.csv")}
    rost = roster()
    xs, ys, cs, labs = [], [], [], []
    for m, r in nomo.items():
        nd = fnum(r.get("nomo_dir_pct")); vf = fnum(lb.get(m, {}).get("var_fidelity"))
        if nd is None or vf is None or vf > 1.0:  # drop degenerate (e.g. qwen3.5-9b vf>1)
            continue
        fam = rost.get(m, {}).get("family", "")
        xs.append(vf); ys.append(nd); cs.append(FAMCOL.get(fam, "#888888")); labs.append(m)
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(xs, ys, c=cs, s=42, edgecolor="white", lw=0.5, zorder=3)
    if len(xs) >= 3:
        b, a = np.polyfit(xs, ys, 1); r = np.corrcoef(xs, ys)[0, 1]
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, a + b*xx, "-", color="#333333", lw=1.3, zorder=2)
        ax.text(0.97, 0.05, f"$r = {r:+.2f}$", transform=ax.transAxes, ha="right", fontsize=9)
    ax.set_xlabel("Within-person variance error\n(lower = closer to human norm)")
    ax.set_ylabel("Nomological direction agreement (%)")
    ax.set_title("Internal vs. external validity")
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_tradeoff.{ext}")
    plt.close(fig); print("wrote fig_tradeoff")

if __name__ == "__main__":
    fig_scaling(); fig_bycriterion(); fig_tradeoff()
