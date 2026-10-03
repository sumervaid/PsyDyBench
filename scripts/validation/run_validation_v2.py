#!/usr/bin/env python3
"""
Internal-validity battery (v2) leaderboard driver.

Runs internal_validity_v2.run_internal_validity_v2 on every model's generated corpus under a
generation run directory and assembles a continuous leaderboard with bootstrap
CIs and equivalence-criterion counts.

Target building (reused from run_validation.py's logic):
  * Per-state grand_mean / total_sd / icc read from data/seeds/norms.json.
  * Between/within SD DERIVED: between = total*sqrt(icc); within = total*sqrt(1-icc).
  * Cross-state correlation + AR(1) targets MERGED from
    data/seeds/cross_state_targets.json if present (entries flagged
    placeholder/unsourced are passed through and treated as TARGET_UNAVAILABLE by
    the battery — never invented).

The merged target dict is written to results/leaderboard/internal_validity_v2_targets.json
so every leaderboard number traces to a saved targets file.

Outputs:
  results/leaderboard/internal_validity_v2_targets.json    (merged targets, provenance)
  results/leaderboard/<MODEL>_internal_validity_v2.json     (full per-model report)
  results/leaderboard/leaderboard_v2.csv          (continuous metrics + CIs + #equiv)

Usage:
  python3 scripts/validation/run_validation_v2.py results/generation/full
  python3 scripts/validation/run_validation_v2.py results/generation/full --n-boot 1000

NOTE: bootstrap with the MixedLM null-model ICC is the slow part. For a quick
structural pass use --n-boot 200; real leaderboard runs use --n-boot 1000.
"""

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import pandas as pd

import internal_validity_v2 as gv2  # same directory


def build_targets(repo: Path, out_dir: Path) -> dict:
    """norms.json (+derived SDs) merged with cross_state_targets.json."""
    norms = json.load(open(repo / "data/seeds/norms.json"))
    states = {}
    for s, st in norms["states"].items():
        icc, tsd = st["icc"], st["total_sd"]
        states[s] = {
            "grand_mean": st["grand_mean"], "total_sd": tsd, "icc": icc,
            "between_person_sd": round(tsd * math.sqrt(icc), 4),
            "within_person_sd": round(tsd * math.sqrt(1 - icc), 4),
        }
    targets = {"states": states,
               "_provenance": "grand_mean/total_sd/icc from data/seeds/norms.json "
                              "(cited); between/within SD derived."}

    cs_path = repo / "data/seeds/cross_state_targets.json"
    if cs_path.exists():
        cs = json.load(open(cs_path))
        # Sourced schema (current): top-level 'pairs' (between-person r's). Legacy
        # schema: nested 'cross_state'/'temporal'. Pass both through; the battery's
        # _cross_state_target handles whichever is present.
        if "pairs" in cs:
            targets["pairs"] = cs["pairs"]
        targets["cross_state"] = cs.get("cross_state", {})
        targets["temporal"] = cs.get("temporal", {})
        targets["_cross_state_provenance"] = cs.get("_meta", {})
    else:
        targets["cross_state"] = {}
        targets["temporal"] = {}
        targets["_cross_state_provenance"] = "cross_state_targets.json absent; "\
            "L3/L4 reported descriptively (TARGET_UNAVAILABLE)."

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "internal_validity_v2_targets.json").write_text(json.dumps(targets, indent=2))
    return targets


def flatten_row(model: str, rep: dict) -> dict:
    """One leaderboard row: continuous fidelity metrics + CIs + #criteria equiv."""
    s = rep["summary"]
    L1, L2, L4 = rep["levels"]["L1"], rep["levels"]["L2"], rep["levels"]["L4"]
    L6 = rep["levels"]["L6"]

    # Mean |Hedges g| across states (lower = closer means).
    gs = [abs(e["hedges_g"]) for e in L1["per_state"].values()
          if e.get("hedges_g") is not None]
    # Mean |log within-SD ratio| across states (the primary variance discriminator).
    import numpy as np
    wlr = []
    for e in L2["per_state"].values():
        eq = e.get("within_sd_equivalence", {})
        if eq.get("ratio"):
            wlr.append(abs(math.log(eq["ratio"])))
    # Mean |ICC - target| across states.
    iccd = []
    for e in L2["per_state"].values():
        if e.get("synth_icc") is not None and e.get("target_icc") not in (None, "",):
            try:
                iccd.append(abs(e["synth_icc"] - float(e["target_icc"])))
            except (TypeError, ValueError):
                pass

    return {
        "model": model,
        "n_personas": rep["n_personas"],
        "n_obs": rep["n_obs"],
        "n_boot": rep["n_boot"],
        # Headline equivalence counts.
        "state_criteria_equivalent": s["n_state_criteria_equivalent"],
        "state_criteria_evaluated": s["n_state_criteria_evaluated"],
        # Continuous distances (lower = better fidelity).
        "mean_abs_hedges_g": round(float(np.mean(gs)), 4) if gs else "",
        "mean_abs_log_within_sd_ratio": round(float(np.mean(wlr)), 4) if wlr else "",
        "mean_abs_icc_diff": round(float(np.mean(iccd)), 4) if iccd else "",
        # L3 nomological (between-person; sourced JSON has no within targets).
        "L3_mean_abs_dr_between": s["L3_mean_abs_dr_between"] if s["L3_mean_abs_dr_between"] is not None else "",
        "L3_pct_pairs_equiv_between": s["L3_pct_pairs_equivalent_between"] if s["L3_pct_pairs_equivalent_between"] is not None else "",
        # Conservative L3 count excluding expert priors (sourced meta-analytic only).
        "L3_pct_pairs_equiv_sourced_only": s.get("L3_pct_pairs_equivalent_sourced_only") if s.get("L3_pct_pairs_equivalent_sourced_only") is not None else "",
        # L4 temporal: mean fixed AR(1) + mean % estimable.
        "L4_mean_ar1": round(float(np.mean([e["ar1_fixed"] for e in L4["per_state"].values()
                                            if e.get("ar1_fixed") is not None])), 4)
        if any(e.get("ar1_fixed") is not None for e in L4["per_state"].values()) else "",
        "L4_mean_pct_estimable": round(float(np.mean([e["pct_personas_estimable"]
                                                      for e in L4["per_state"].values()
                                                      if e.get("pct_personas_estimable") is not None])), 1)
        if any(e.get("pct_personas_estimable") is not None for e in L4["per_state"].values()) else "",
        # L6 structure (descriptive).
        "L6_between_pc1": L6["between_person"]["pc1_var_explained"] if L6["between_person"]["pc1_var_explained"] is not None else "",
        "L6_within_pc1": L6["within_person"]["pc1_var_explained"] if L6["within_person"]["pc1_var_explained"] is not None else "",
        # L7 pileup count.
        "L7_states_with_pileup": rep["levels"]["L7"]["n_states_with_pileup"],
    }


def main():
    ap = argparse.ArgumentParser(description="Internal-validity battery (v2) leaderboard across a generation run.")
    ap.add_argument("base", nargs="?", default="results/generation/full",
                    help="Generation run dir containing <MODEL>/states.csv.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    base = Path(args.base)
    if not base.is_absolute():
        base = repo / base
    out_dir = repo / "results/leaderboard"

    targets = build_targets(repo, out_dir)

    board = []
    csvs = sorted(glob.glob(str(base / "*/states.csv")))
    if not csvs:
        # smoke run stores states.csv directly under the run dir.
        if (base / "states.csv").exists():
            csvs = [str(base / "states.csv")]
    print(f"Found {len(csvs)} corpora under {base}")

    for sc in csvs:
        model = Path(sc).parent.name
        print(f"  [{model}] ...", flush=True)
        df = pd.read_csv(sc)
        rep = gv2.run_internal_validity_v2(df, targets, n_boot=args.n_boot, seed=args.seed)
        json.dump(rep, open(out_dir / f"{model}_internal_validity_v2.json", "w"),
                  indent=2, cls=gv2.NumpyEncoder)
        board.append(flatten_row(model, rep))

    # Sort by best variance fidelity then closest means.
    def sortkey(r):
        v = r["mean_abs_log_within_sd_ratio"]
        g = r["mean_abs_hedges_g"]
        return (v if isinstance(v, (int, float)) else 9,
                g if isinstance(g, (int, float)) else 9)
    board.sort(key=sortkey)

    cols = ["model", "n_personas", "n_obs", "state_criteria_equivalent",
            "state_criteria_evaluated", "mean_abs_hedges_g",
            "mean_abs_log_within_sd_ratio", "mean_abs_icc_diff",
            "L3_mean_abs_dr_between", "L3_pct_pairs_equiv_between",
            "L3_pct_pairs_equiv_sourced_only",
            "L4_mean_ar1", "L4_mean_pct_estimable",
            "L6_between_pc1", "L6_within_pc1", "L7_states_with_pileup", "n_boot"]
    out_csv = out_dir / "leaderboard_v2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in board:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"\nWrote {out_csv}")
    print(f"{'model':30s} {'equiv':>6s} {'|g|':>6s} {'|lnWSDr|':>8s} {'AR1':>6s} {'%est':>6s}")
    for r in board:
        def fmt(x, p="{:.3f}"):
            return p.format(x) if isinstance(x, (int, float)) else "   -"
        print(f"{r['model']:30s} "
              f"{r['state_criteria_equivalent']:>2}/{r['state_criteria_evaluated']:<3} "
              f"{fmt(r['mean_abs_hedges_g']):>6s} {fmt(r['mean_abs_log_within_sd_ratio']):>8s} "
              f"{fmt(r['L4_mean_ar1']):>6s} {fmt(r['L4_mean_pct_estimable'],'{:.0f}'):>6s}")
    print("\nLower |g| and |lnWSDr| = closer to literature. L3 between TARGET_UNAVAILABLE "
          "until a sourced state-correlation matrix is added; observed r's + CIs are in "
          "each <MODEL>_internal_validity_v2.json.")


if __name__ == "__main__":
    main()
