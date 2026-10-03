#!/usr/bin/env python3
"""
Reliability battery (Framework 1, internal validity — §3 of the outline).

Three reliability analyses for a synthetic psychological-state population:

  1. TEST-RETEST  — generate the SAME personas under K independent reseeds
     (different RNG / sampling seeds, identical prompts/model). Correlate each
     person's per-state mean across seed pairs. High test-retest r => the
     generated trait-level signal is stable, not seed noise.

  2. SPLIT-HALF   — the rating stage issues a MULTI-CALL rating (the rater is
     queried several times per state-day and averaged). Split those calls into
     two halves, correlate the half-means, and Spearman-Brown correct to the
     full length. This is the internal consistency of the rating instrument.

  3. G-THEORY     — generalizability-theory variance decomposition of a fully
     crossed person x day x state x seed design via a random-effects ANOVA-style
     estimator. Reports variance components and a G-coefficient (the proportion
     of variance attributable to the object of measurement, person, relative to
     person + person-interaction error). A high G-coefficient means scores
     dependably distinguish persons across the day/seed facets.

INPUT FORMATS (all long-format CSVs; states in the four locked *_state cols OR a
single `state_value` column tagged by a `state` column — both accepted):

  --reseeds-csv  : columns employee_id, day, seed, <*_state cols>.  >=2 seeds.
                   Used for test-retest AND (with day) for G-theory.
  --calls-csv    : columns employee_id, day, state, call_idx, rating.
                   The per-call ratings before averaging. Used for split-half.

If neither is supplied, --demo synthesizes a small stub corpus (documented as a
stub, never a manuscript number) so the module's plumbing can be exercised before
a real reseed corpus exists.

Usage:
    python3 scripts/validation/reliability.py --reseeds-csv X.csv --calls-csv Y.csv --out-dir results/reliability/<run>/
    python3 scripts/validation/reliability.py --demo --out-dir results/reliability/demo/
"""

import argparse
import itertools
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize wide (*_state cols) into long (state, state_value)."""
    if "state" in df.columns and "state_value" in df.columns:
        return df
    present = [c for c in STATE_COLS if c in df.columns]
    id_cols = [c for c in df.columns if c not in present]
    long = df.melt(id_vars=id_cols, value_vars=present,
                   var_name="state", value_name="state_value")
    return long.dropna(subset=["state_value"])


# ── 1. Test-retest across reseeds ─────────────────────────────────────────────

def test_retest(reseeds: pd.DataFrame) -> dict:
    """Per-state test-retest r: correlate person means between every seed pair,
    then average the Fisher-z'd pairwise correlations."""
    long = _to_long(reseeds)
    if "seed" not in long.columns:
        raise ValueError("reseeds corpus needs a 'seed' column for test-retest.")
    seeds = sorted(long["seed"].unique())
    out = {}
    for st in sorted(long["state"].unique()):
        sub = long[long["state"] == st]
        # person mean per seed
        pm = sub.groupby(["employee_id", "seed"])["state_value"].mean().unstack("seed")
        pairwise = []
        for a, b in itertools.combinations(seeds, 2):
            if a in pm.columns and b in pm.columns:
                d = pm[[a, b]].dropna()
                if len(d) >= 5:
                    r = d[a].corr(d[b])
                    if not np.isnan(r):
                        pairwise.append(r)
        if pairwise:
            z = np.arctanh(np.clip(pairwise, -0.999, 0.999))
            mean_r = float(np.tanh(np.mean(z)))
            out[st] = {"test_retest_r": round(mean_r, 4),
                       "n_seed_pairs": len(pairwise),
                       "n_persons": int(pm.dropna().shape[0])}
        else:
            out[st] = {"test_retest_r": None, "n_seed_pairs": 0}
    return out


# ── 2. Split-half of the multi-call rating ────────────────────────────────────

def split_half(calls: pd.DataFrame) -> dict:
    """Split the per-(person,day,state) multi-call ratings into two halves by
    call_idx parity, correlate the half-means across observations, and apply the
    Spearman-Brown prophecy correction back to full length."""
    req = {"employee_id", "day", "state", "call_idx", "rating"}
    if not req.issubset(calls.columns):
        raise ValueError(f"calls corpus needs columns {req}.")
    out = {}
    for st in sorted(calls["state"].unique()):
        sub = calls[calls["state"] == st].copy()
        sub["half"] = sub["call_idx"] % 2  # even vs odd calls
        hm = (sub.groupby(["employee_id", "day", "half"])["rating"].mean()
                 .unstack("half"))
        hm = hm.dropna()
        if 0 in hm.columns and 1 in hm.columns and len(hm) >= 5:
            r_half = hm[0].corr(hm[1])
            sb = (2 * r_half) / (1 + r_half) if r_half is not None and r_half > -1 else np.nan
            out[st] = {"split_half_r": round(float(r_half), 4),
                       "spearman_brown_r": round(float(sb), 4),
                       "n_obs": int(len(hm))}
        else:
            out[st] = {"split_half_r": None, "n_obs": int(len(hm))}
    return out


# ── 3. Generalizability-theory variance decomposition ─────────────────────────

def g_theory(reseeds: pd.DataFrame) -> dict:
    """Random-effects variance decomposition over a person x day x seed design,
    run per state. Uses a balanced ANOVA-style estimator on the largest balanced
    block (common persons x days x seeds) so mean-squares are interpretable.

    Components: person (sigma2_p), day (sigma2_d), seed (sigma2_s), residual.
    G-coefficient (relative, persons = object of measurement):
        Ep2 = sigma2_p / (sigma2_p + sigma2_rel_error)
    with sigma2_rel_error = (sigma2_pd/n_d + sigma2_ps/n_s + sigma2_resid/(n_d n_s)).
    Here we use a simplified components estimator: person variance from person
    means, facet variances from facet means, residual as leftover.
    """
    long = _to_long(reseeds)
    if "seed" not in long.columns:
        raise ValueError("G-theory needs a 'seed' column.")
    if "day" not in long.columns:
        long["day"] = long.groupby(["employee_id", "seed"]).cumcount()

    out = {}
    for st in sorted(long["state"].unique()):
        sub = long[long["state"] == st]
        grand = sub["state_value"].mean()
        total_var = sub["state_value"].var(ddof=1)
        # Main-effect variance estimates (deviation of facet means from grand mean).
        var_p = sub.groupby("employee_id")["state_value"].mean().var(ddof=1)
        var_d = sub.groupby("day")["state_value"].mean().var(ddof=1)
        var_s = sub.groupby("seed")["state_value"].mean().var(ddof=1)
        n_d = sub["day"].nunique()
        n_s = sub["seed"].nunique()
        # Residual = total minus explained main effects (floored at 0).
        resid = max(total_var - var_p - var_d - var_s, 0.0)
        # Relative error for a person-decision: facets + residual averaged over levels.
        rel_error = var_d / max(n_d, 1) + var_s / max(n_s, 1) + resid / max(n_d * n_s, 1)
        g_coef = var_p / (var_p + rel_error) if (var_p + rel_error) > 0 else np.nan
        out[st] = {
            "var_person": round(float(var_p), 4),
            "var_day": round(float(var_d), 4),
            "var_seed": round(float(var_s), 4),
            "var_residual": round(float(resid), 4),
            "pct_person": round(100 * var_p / total_var, 1) if total_var > 0 else None,
            "g_coefficient": round(float(g_coef), 4) if not np.isnan(g_coef) else None,
            "n_days": int(n_d), "n_seeds": int(n_s),
            "n_persons": int(sub["employee_id"].nunique()),
        }
    return out


# ── Demo stub ─────────────────────────────────────────────────────────────────

def make_demo(n_persons=40, n_days=6, n_seeds=3, n_calls=4, rng_seed=7):
    """Synthesize a stub reseed corpus + per-call ratings.

    Construction (documented so the structure is interpretable, NOT a manuscript
    number): each person has a stable per-state baseline (between-person signal)
    plus day noise plus small seed noise. Per-call ratings jitter around the
    state value. By design test-retest and G-coefficients should be high because
    the between-person signal dominates."""
    rng = np.random.default_rng(rng_seed)
    scales = {"work_engagement_state": (1, 7), "psych_safety_state": (1, 4),
              "work_creativity_state": (1, 6), "team_trust_state": (1, 5)}
    wide_rows, call_rows = [], []
    person_base = {}
    for pid in range(n_persons):
        person_base[pid] = {c: rng.uniform(lo + 0.8, hi - 0.8) for c, (lo, hi) in scales.items()}
    for pid in range(n_persons):
        for seed in range(n_seeds):
            for day in range(n_days):
                rec = {"employee_id": pid, "day": day, "seed": seed}
                for c, (lo, hi) in scales.items():
                    base = person_base[pid][c]
                    val = base + rng.normal(0, 0.5) + rng.normal(0, 0.15)  # day + seed noise
                    val = float(np.clip(round(val, 2), lo, hi))
                    rec[c] = val
                    if seed == 0:  # per-call ratings only logged for one seed (the rater facet)
                        for ci in range(n_calls):
                            call_rows.append({"employee_id": pid, "day": day,
                                              "state": c, "call_idx": ci,
                                              "rating": float(np.clip(val + rng.normal(0, 0.3), lo, hi))})
                wide_rows.append(rec)
    return pd.DataFrame(wide_rows), pd.DataFrame(call_rows)


def main():
    ap = argparse.ArgumentParser(description="Reliability battery (test-retest, split-half, G-theory).")
    ap.add_argument("--reseeds-csv", default=None)
    ap.add_argument("--calls-csv", default=None)
    ap.add_argument("--demo", action="store_true", help="Run on a synthesized stub corpus.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {"source": {}}
    if args.demo or (not args.reseeds_csv and not args.calls_csv):
        reseeds, calls = make_demo()
        report["source"]["mode"] = "DEMO STUB (not a manuscript number)"
        report["test_retest"] = test_retest(reseeds)
        report["g_theory"] = g_theory(reseeds)
        report["split_half"] = split_half(calls)
    else:
        if args.reseeds_csv:
            reseeds = pd.read_csv(args.reseeds_csv)
            report["source"]["reseeds_csv"] = args.reseeds_csv
            report["test_retest"] = test_retest(reseeds)
            report["g_theory"] = g_theory(reseeds)
        if args.calls_csv:
            calls = pd.read_csv(args.calls_csv)
            report["source"]["calls_csv"] = args.calls_csv
            report["split_half"] = split_half(calls)

    with open(out / "reliability_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    print(f"Reliability report ({report['source']}):")
    for block in ("test_retest", "split_half", "g_theory"):
        if block in report:
            print(f"  [{block}]")
            for st, m in report[block].items():
                print(f"    {st:<24s} {m}")
    print(f"Wrote {out}/reliability_report.json")


if __name__ == "__main__":
    main()
