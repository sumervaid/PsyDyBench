#!/usr/bin/env python3
"""
Meta-analytic convergence anchoring — repointed to a GENERATED-states corpus
(Framework 2, external validity).

The existing `meta_analysis_validation.py` correlates the 55 inverse-engine
indicators against the REAL ESM states baked into `inverse_corpus_prepared.csv`
(model-independent — it validates the inverse engine, not any forward corpus).

This module reuses the SAME anchor library (`META_BENCHMARKS`, 118 pairs) and the
SAME correlation estimators (`compute_pooled_r`, `compute_within_r`) but lets you
point the STATE side at any generated corpus via `--states-csv`, joining on
(employee_id, day) exactly like `scripts/analysis/per_model_recovery_phase14.py`.
The INDICATOR side comes from `--indicators-csv` (the daily-context matrix). For
each anchored (indicator, state) pair it reports observed pooled & within-person r
against the published meta-analytic r, then summarizes direction-match % and
signed Spearman rho (pooled and within) — the convergence metric.

This is the single knob D3 asked for: anchoring that runs on generated states
rather than only the real corpus, with the anchor table single-sourced from
`meta_analysis_validation.py` (no fork of the 118 values).

INPUTS
  --indicators-csv : daily-context indicator matrix (employee_id, day, <55 cols>).
                     For the all-silico paper this is the seeded/forward indicator
                     matrix; for testing it can be the real inverse corpus.
  --states-csv     : generated-states corpus (employee_id, day, *_state).
                     If omitted, states are taken from --indicators-csv itself
                     (i.e. fall back to the in-file real states; this reproduces
                     the legacy single-file behavior).

OUTPUTS (under --out-dir)
  anchoring_pairs.csv   : per-pair indicator,state,meta_r,pooled_r,within_r,dir_*
  anchoring_summary.csv : one summary row (n_pairs, dirmatch%, rho) pooled+within
  RESULTS_anchoring.md  : human-readable + provenance

Usage:
    python3 scripts/validation/anchoring.py \
        --indicators-csv data/seeds/forward_indicators.csv \
        --states-csv     results/run/generated_states.csv \
        --out-dir        results/anchoring/<run>/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Single-source the anchor library + estimators (no fork of the 118 values).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_analysis_validation import (  # noqa: E402
    META_BENCHMARKS,
    compute_pooled_r,
    compute_within_r,
    DIM_PREFIXES,
)

STATE_COLS = [
    "work_engagement_state",
    "psych_safety_state",
    "work_creativity_state",
    "team_trust_state",
]


def _indicator_columns(df: pd.DataFrame):
    """Keep join keys + true indicator columns; drop any *_state columns so the
    indicator frame can never accidentally carry real states into the join."""
    keep = []
    for c in df.columns:
        if c in ("employee_id", "day"):
            keep.append(c)
        elif c in STATE_COLS:
            continue
        elif c.split("_")[0] in DIM_PREFIXES or c.split("_")[0] == "life":
            keep.append(c)
    return df[keep]


def build_merged(indicators_csv: Path, states_csv) -> pd.DataFrame:
    """Join the indicator matrix to the states corpus on (employee_id, day)."""
    ind = pd.read_csv(indicators_csv)
    ind = _indicator_columns(ind)
    if "day" not in ind.columns:
        ind["day"] = ind.groupby("employee_id").cumcount()

    if states_csv is None:
        # Legacy single-file mode: states live in the indicator file itself.
        full = pd.read_csv(indicators_csv)
        keep = ["employee_id", "day"] + [c for c in STATE_COLS if c in full.columns]
        if "day" not in full.columns:
            full["day"] = full.groupby("employee_id").cumcount()
            keep = ["employee_id", "day"] + [c for c in STATE_COLS if c in full.columns]
        states = full[keep]
    else:
        states = pd.read_csv(states_csv)
        if "day" not in states.columns:
            states["day"] = states.groupby("employee_id").cumcount()
        states = states[["employee_id", "day"] + [c for c in STATE_COLS if c in states.columns]]

    return ind.merge(states, on=["employee_id", "day"], how="inner")


def compute_pairs(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entry in META_BENCHMARKS:
        ind, st, meta_r = entry["indicator"], entry["state"], entry["meta_r"]
        if ind not in merged.columns or st not in merged.columns:
            continue
        pooled = compute_pooled_r(merged, ind, st)
        within = compute_within_r(merged, ind, st)
        rows.append({
            "indicator": ind, "state": st, "meta_r": meta_r,
            "pooled_r": pooled, "within_r": within,
            "dir_pooled": float(np.sign(pooled) == np.sign(meta_r)) if pd.notna(pooled) else np.nan,
            "dir_within": float(np.sign(within) == np.sign(meta_r)) if pd.notna(within) else np.nan,
            "confidence": entry.get("confidence", ""),
            "citation": entry.get("citation", ""),
        })
    return pd.DataFrame(rows)


def summarize(pairs: pd.DataFrame, merged: pd.DataFrame) -> dict:
    vp = pairs.dropna(subset=["pooled_r"])
    vw = pairs.dropna(subset=["within_r"])

    def signed_rho(sub, col):
        if len(sub) < 3:
            return np.nan
        return stats.spearmanr(sub[col], sub["meta_r"]).correlation

    return {
        "n_obs": int(len(merged)),
        "n_employees": int(merged["employee_id"].nunique()),
        "n_pairs_pooled": int(len(vp)),
        "dirmatch_pooled_pct": round(100 * vp["dir_pooled"].mean(), 2) if len(vp) else np.nan,
        "rho_pooled": round(signed_rho(vp, "pooled_r"), 4) if len(vp) else np.nan,
        "n_pairs_within": int(len(vw)),
        "dirmatch_within_pct": round(100 * vw["dir_within"].mean(), 2) if len(vw) else np.nan,
        "rho_within": round(signed_rho(vw, "within_r"), 4) if len(vw) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="Meta-analytic anchoring on generated states.")
    ap.add_argument("--indicators-csv", required=True, help="Indicator matrix CSV (employee_id, day, 55 cols).")
    ap.add_argument("--states-csv", default=None, help="Generated-states corpus CSV. Omit = use states in --indicators-csv.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--label", default="run", help="Run label for the markdown header.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged = build_merged(Path(args.indicators_csv), args.states_csv)
    pairs = compute_pairs(merged)
    pairs.to_csv(out / "anchoring_pairs.csv", index=False)
    summ = summarize(pairs, merged)
    pd.DataFrame([summ]).to_csv(out / "anchoring_summary.csv", index=False)

    src = args.states_csv if args.states_csv else f"{args.indicators_csv} (in-file states)"
    print(f"Anchoring [{args.label}]  states={src}")
    print(f"  N={summ['n_employees']} emp / {summ['n_obs']} obs")
    print(f"  pooled : {summ['n_pairs_pooled']} pairs  dir={summ['dirmatch_pooled_pct']}%  rho={summ['rho_pooled']}")
    print(f"  within : {summ['n_pairs_within']} pairs  dir={summ['dirmatch_within_pct']}%  rho={summ['rho_within']}")

    with open(out / "RESULTS_anchoring.md", "w") as fh:
        fh.write(f"# Meta-analytic convergence anchoring — {args.label}\n\n")
        fh.write("Generated by `scripts/validation/anchoring.py`.\n\n")
        fh.write(f"- Indicators: `{args.indicators_csv}`\n")
        fh.write(f"- States: `{src}`\n")
        fh.write("- Anchors: `META_BENCHMARKS` (118 pairs) from `scripts/validation/meta_analysis_validation.py`.\n")
        fh.write("- Convergence = direction-match %% and signed Spearman rho of observed r vs published meta_r, "
                 "computed POOLED (between-person dominated) and WITHIN-person.\n\n")
        fh.write("| metric | pooled | within |\n|---|---|---|\n")
        fh.write(f"| n pairs | {summ['n_pairs_pooled']} | {summ['n_pairs_within']} |\n")
        fh.write(f"| direction match %% | {summ['dirmatch_pooled_pct']} | {summ['dirmatch_within_pct']} |\n")
        fh.write(f"| Spearman rho | {summ['rho_pooled']} | {summ['rho_within']} |\n")
        fh.write(f"\nN = {summ['n_employees']} employees, {summ['n_obs']} observations.\n")
    print(f"Wrote {out}/anchoring_pairs.csv, anchoring_summary.csv, RESULTS_anchoring.md")


if __name__ == "__main__":
    main()
