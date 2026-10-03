#!/usr/bin/env python3
"""
Internal-validity battery, v1: the internal-structure validation battery (Framework 1).

Self-contained, pure-statistics re-implementation of the earlier stage-4 validator
(`synthetic_corpus_generator_v5/stages/stage4_validation.py`) and the
DSPy internal-structure metric (an earlier DSPy metric), repointed so it
runs on ANY generated-corpus CSV given a target-norms JSON. No LLM, no DSPy, no
import of the genagents monorepo — only pandas/numpy/scipy.

The six levels (PA/NA level 5 permanently removed from this project):
  L1 Distributions   — grand mean & SD vs target (per state)
  L2 ICC / variance  — between/within variance ratio vs target ICC
  L3 Cross-state     — between- & within-person state correlations vs target matrices
  L4 Temporal        — mean per-person lag-1 autocorrelation in a plausible band
  L6 Factor          — average inter-state correlation in a moderate band
  L7 Range/extremes  — scale coverage and ceiling/floor pileup

Mirrors the same thresholds and scoring as the monorepo validator so numbers are
comparable to the saved Phase-14 `validation_report.json` files.

TARGET NORMS (JSON) — two accepted shapes, both handled:

  Preferred ("states" shape):
    {
      "states": {
        "work_engagement": {"grand_mean": 4.5, "grand_sd": 1.37, "icc": 0.748,
                             "between_person_sd": 1.20, "within_person_sd": 0.70},
        ... (one block per state) ...
      },
      "state_col_order": ["work_engagement_state", "psych_safety_state",
                          "work_creativity_state", "team_trust_state"],
      "between_person_state_corr": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "within_person_state_corr":  [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }

  Legacy ("corpus_targets" shape) is also normalized (same logic as the
  monorepo `load_targets`).

CORPUS CSV — must contain `employee_id`, `day`, and the four *_state columns
(work_engagement_state, psych_safety_state, work_creativity_state,
team_trust_state). Missing states are skipped, not fatal.

Usage:
    python3 scripts/validation/internal_validity.py \
        --corpus path/to/corpus.csv \
        --targets path/to/psychometric_targets.json \
        --out results/leaderboard/validation_report.json

If --targets is omitted, a NEUTRAL self-referential target is built from the
corpus itself (means/SDs/ICC become their own targets, zero target correlations)
so L1/L2/L3 still execute and report observed values. This is only for smoke
testing the module's plumbing; real runs MUST pass --targets.
"""

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── State definitions (locked; do NOT re-derive) ─────────────────────────────
# name -> (state_col, scale_min, scale_max)
STATES: Dict[str, Dict[str, Any]] = {
    "work_engagement": {"col": "work_engagement_state", "min": 1.0, "max": 7.0},
    "psych_safety":    {"col": "psych_safety_state",    "min": 1.0, "max": 4.0},
    "work_creativity": {"col": "work_creativity_state", "min": 1.0, "max": 6.0},
    "team_trust":      {"col": "team_trust_state",       "min": 1.0, "max": 5.0},
}
STATE_NAMES = list(STATES.keys())
STATE_COLS = [STATES[s]["col"] for s in STATE_NAMES]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that survives numpy scalar/array/bool types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


@dataclass
class ValidationLevel:
    level: int
    name: str
    passed: bool
    score: float
    metrics: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Target loading / normalization ───────────────────────────────────────────

def load_targets(path: Path) -> dict:
    """Load a target-norms JSON, normalizing the legacy corpus_targets shape."""
    with open(path) as f:
        raw = json.load(f)

    if "corpus_targets" in raw and "states" not in raw:
        raw["states"] = raw["corpus_targets"]
        bc = raw.get("between_correlations")
        if isinstance(bc, dict) and "matrix" in bc:
            raw["between_person_state_corr"] = bc["matrix"]
            raw.setdefault("state_col_order", bc.get("columns", []))
        wc = raw.get("within_correlations")
        if isinstance(wc, dict) and "matrix" in wc:
            raw["within_person_state_corr"] = wc["matrix"]
    return raw


def neutral_targets_from_corpus(df: pd.DataFrame) -> dict:
    """Build a self-referential target (corpus is its own target).

    Means/SDs/ICC are read off the corpus; target correlations are zero. Used
    ONLY for smoke testing the module so L1/L2 pass trivially and L3 reports the
    observed correlations as deviations from zero. Real runs pass --targets.
    """
    states: Dict[str, Any] = {}
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        icc, b_sd, w_sd = _compute_icc(df, col)
        states[name] = {
            "grand_mean": float(vals.mean()),
            "grand_sd": float(vals.std()),
            "icc": float(icc) if not np.isnan(icc) else 0.5,
            "between_person_sd": float(b_sd) if not np.isnan(b_sd) else 0.0,
            "within_person_sd": float(w_sd) if not np.isnan(w_sd) else 0.0,
        }
    zero = [[0.0] * len(STATE_COLS) for _ in STATE_COLS]
    return {
        "states": states,
        "state_col_order": STATE_COLS,
        "between_person_state_corr": zero,
        "within_person_state_corr": zero,
        "_neutral_self_referential": True,
    }


# ── Shared statistics ─────────────────────────────────────────────────────────

def _compute_icc(df: pd.DataFrame, col: str):
    """Return (icc, between_sd, within_sd) using the same pooled-within estimator
    as the monorepo stage-4 validator (variance of person means over total var)."""
    if col not in df.columns:
        return np.nan, np.nan, np.nan
    g = df.groupby("employee_id")[col].agg(["mean", "std", "count"])
    g = g[g["count"] >= 2]
    if len(g) < 5:
        return np.nan, np.nan, np.nan
    between_var = g["mean"].var()
    g = g.assign(within_var=g["std"] ** 2, weight=g["count"] - 1).dropna()
    if g["weight"].sum() > 0:
        pooled_within = (g["within_var"] * g["weight"]).sum() / g["weight"].sum()
    else:
        pooled_within = 0.0
    total = between_var + pooled_within
    icc = between_var / total if total > 0 else np.nan
    return icc, np.sqrt(between_var), np.sqrt(pooled_within)


def _target_r(matrix, order, col1, col2) -> float:
    if not matrix or not order or col1 not in order or col2 not in order:
        return 0.0
    i, j = order.index(col1), order.index(col2)
    if i < len(matrix) and j < len(matrix[i]):
        return float(matrix[i][j])
    return 0.0


# ── Levels ────────────────────────────────────────────────────────────────────

def level1_distributions(df, targets) -> ValidationLevel:
    metrics, recs, checks = {}, [], []
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        tgt = targets["states"].get(name, {})
        s_mean, s_sd = vals.mean(), vals.std()
        t_mean = tgt.get("grand_mean", s_mean)
        t_sd = tgt.get("grand_sd", s_sd)
        mean_diff = abs(s_mean - t_mean)
        mean_ok = mean_diff < 0.5
        sd_ratio = s_sd / t_sd if t_sd > 0 else 1.0
        sd_ok = 0.7 < sd_ratio < 1.3
        metrics[name] = {
            "synth_mean": round(float(s_mean), 3), "target_mean": round(float(t_mean), 3),
            "mean_diff": round(float(mean_diff), 3), "mean_ok": bool(mean_ok),
            "synth_sd": round(float(s_sd), 3), "target_sd": round(float(t_sd), 3),
            "sd_ratio": round(float(sd_ratio), 3), "sd_ok": bool(sd_ok),
        }
        checks.append(mean_ok and sd_ok)
        if not sd_ok and sd_ratio < 0.7:
            recs.append(f"{name}: under-variance (SD ratio {sd_ratio:.2f}); add vivid extremes to identities.")
    score = sum(checks) / len(checks) if checks else 0.0
    return ValidationLevel(1, "Distributions", score >= 0.75, score, metrics, recs)


def level2_icc(df, targets) -> ValidationLevel:
    metrics, recs, checks = {}, [], []
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        icc, b_sd, w_sd = _compute_icc(df, col)
        if np.isnan(icc):
            metrics[name] = {"error": "not enough employees with >=2 days"}
            continue
        tgt = targets["states"].get(name, {})
        t_icc = tgt.get("icc", 0.5)
        t_wsd = tgt.get("within_person_sd", 0) or 0
        icc_diff = abs(icc - t_icc)
        icc_ok = icc_diff < 0.15
        # GRADED score on within-person-SD closeness (the discriminating signal that the old
        # binary pass/fail collapsed): 1.0 = exact; decays with |ln(synth_wsd / target_wsd)|,
        # reaching ~0 at e^±1.5 (~4.5x off). Falls back to graded ICC closeness if no target SD.
        if t_wsd > 0 and w_sd > 0:
            state_score = max(0.0, 1.0 - abs(float(np.log(w_sd / t_wsd))) / 1.5)
        else:
            state_score = max(0.0, 1.0 - icc_diff / 0.30)
        metrics[name] = {
            "synth_icc": round(float(icc), 3), "target_icc": round(float(t_icc), 3),
            "icc_diff": round(float(icc_diff), 3), "icc_ok": bool(icc_ok),
            "synth_within_sd": round(float(w_sd), 3), "target_within_sd": round(float(t_wsd), 3),
            "synth_between_sd": round(float(b_sd), 3),
            "state_score": round(float(state_score), 3),
        }
        checks.append(state_score)
        if not icc_ok:
            direction = "too trait-like" if icc > t_icc else "too state-like (over-varying)"
            recs.append(f"{name}: {direction} (within-SD {w_sd:.2f} vs target {t_wsd:.2f}).")
    score = sum(checks) / len(checks) if checks else 0.0
    return ValidationLevel(2, "ICC (Variance Ratios)", score >= 0.75, score, metrics, recs)


def level3_correlations(df, targets) -> ValidationLevel:
    order = targets.get("state_col_order", STATE_COLS)
    bmat = targets.get("between_person_state_corr", [])
    wmat = targets.get("within_person_state_corr", [])
    present = [n for n in STATE_NAMES if STATES[n]["col"] in df.columns]
    cols = [STATES[n]["col"] for n in present]

    emp_means = df.groupby("employee_id")[cols].mean()
    between_corr = emp_means.corr()

    dev = df.copy()
    for c in cols:
        dev[f"{c}_dev"] = dev[c] - dev.groupby("employee_id")[c].transform("mean")
    within_corr = dev[[f"{c}_dev" for c in cols]].corr()

    metrics: Dict[str, Any] = {"between_person": {}, "within_person": {}}
    checks, recs = [], []
    for i, s1 in enumerate(present):
        for j, s2 in enumerate(present):
            if i >= j:
                continue
            c1, c2 = STATES[s1]["col"], STATES[s2]["col"]
            key = f"{s1}_x_{s2}"
            # between
            b = float(between_corr.loc[c1, c2]) if c1 in between_corr.index else 0.0
            tb = _target_r(bmat, order, c1, c2)
            metrics["between_person"][key] = {
                "synth_r": round(b, 3), "target_r": round(tb, 3),
                "diff": round(abs(b - tb), 3), "ok": abs(b - tb) < 0.2}
            checks.append(abs(b - tb) < 0.2)
            # within
            w = float(within_corr.loc[f"{c1}_dev", f"{c2}_dev"])
            tw = _target_r(wmat, order, c1, c2)
            metrics["within_person"][key] = {
                "synth_r": round(w, 3), "target_r": round(tw, 3),
                "diff": round(abs(w - tw), 3), "ok": abs(w - tw) < 0.2}
            checks.append(abs(w - tw) < 0.2)
            if abs(w - tw) > 0.3 or abs(b - tb) > 0.3:
                recs.append(f"Correlation mismatch on {key}; add state-relationship cues to identities.")
    score = sum(checks) / len(checks) if checks else 0.0
    return ValidationLevel(3, "State Correlations", score >= 0.6, score, metrics, recs)


def level4_temporal(df) -> ValidationLevel:
    metrics, checks = {}, []
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        acs = []
        for _, g in df.groupby("employee_id"):
            v = g.sort_values("day")[col].dropna().values
            if len(v) < 3 or np.std(v[:-1]) < 1e-9 or np.std(v[1:]) < 1e-9:
                continue  # constant series -> AR(1) undefined
            r = np.corrcoef(v[:-1], v[1:])[0, 1]
            if not np.isnan(r):
                acs.append(r)
        if acs:
            m = float(np.mean(acs))
            ok = -0.2 < m < 0.6
            metrics[name] = {"mean_ar1": round(m, 3), "n_employees": len(acs), "ok": bool(ok)}
            checks.append(ok)
        else:
            metrics[name] = {"error": "could not compute AR(1)"}
    score = sum(checks) / len(checks) if checks else 0.5
    return ValidationLevel(4, "Temporal Patterns", score >= 0.5, score, metrics, [])


def level6_factor(df) -> ValidationLevel:
    cols = [c for c in STATE_COLS if c in df.columns]
    if len(cols) < 3:
        return ValidationLevel(6, "Factor Structure", True, 0.5,
                               {"error": "not enough states"}, [])
    corr = df[cols].corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    avg = float(upper.stack().mean())
    ok = 0.1 < avg < 0.7
    recs = []
    if avg > 0.7:
        recs.append("States too highly correlated; reduce lock-step movement.")
    return ValidationLevel(6, "Factor Structure", ok, 1.0 if ok else 0.3,
                           {"avg_inter_state_correlation": round(avg, 3)}, recs)


def level7_patterns(df) -> ValidationLevel:
    metrics, recs, checks = {}, [], []
    for name in STATE_NAMES:
        col, smin, smax = STATES[name]["col"], STATES[name]["min"], STATES[name]["max"]
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        at_min = (vals <= smin + 0.1).sum() / len(vals)
        at_max = (vals >= smax - 0.1).sum() / len(vals)
        range_used = (vals.max() - vals.min()) / (smax - smin) if smax > smin else 0.0
        range_ok = range_used > 0.4
        metrics[name] = {
            "pct_at_min": round(at_min * 100, 1), "pct_at_max": round(at_max * 100, 1),
            "range_used": round(float(range_used), 3),
            "min_observed": round(float(vals.min()), 2), "max_observed": round(float(vals.max()), 2),
            "extremes_ok": bool(at_min < 0.15 and at_max < 0.15), "range_ok": bool(range_ok),
        }
        checks.append(range_ok)
        if not range_ok:
            recs.append(f"{name}: only {range_used*100:.0f}% of scale used; increase variance.")
    score = sum(checks) / len(checks) if checks else 0.5
    return ValidationLevel(7, "Patterns & Extremes", score >= 0.75, score, metrics, recs)


# ── Driver ────────────────────────────────────────────────────────────────────

def run_internal_validity(df: pd.DataFrame, targets: dict) -> dict:
    levels = [
        level1_distributions(df, targets),
        level2_icc(df, targets),
        level3_correlations(df, targets),
        level4_temporal(df),
        level6_factor(df),
        level7_patterns(df),
    ]
    passed = sum(1 for lv in levels if lv.passed)
    total = len(levels)
    overall = float(np.mean([lv.score for lv in levels]))
    if passed >= total - 1:
        summary = f"GOOD: {passed}/{total} levels passed."
    elif passed >= total // 2:
        summary = f"MODERATE: {passed}/{total} levels passed; identity refinement needed."
    else:
        summary = f"NEEDS WORK: only {passed}/{total} levels passed."
    recs: List[str] = []
    for lv in levels:
        recs.extend(lv.recommendations)
    return {
        "levels_passed": passed,
        "total_levels": total,
        "overall_score": round(overall, 3),
        "n_obs": int(len(df)),
        "n_employees": int(df["employee_id"].nunique()),
        "levels": [lv.to_dict() for lv in levels],
        "summary": summary,
        "recommendations": recs[:10],
        "neutral_targets": bool(targets.get("_neutral_self_referential", False)),
    }


def main():
    ap = argparse.ArgumentParser(description="Internal-validity battery (v1) on a generated corpus.")
    ap.add_argument("--corpus", required=True, help="Corpus CSV (employee_id, day, *_state).")
    ap.add_argument("--targets", default=None, help="Target-norms JSON. Omit to self-reference (smoke test only).")
    ap.add_argument("--out", default=None, help="Output JSON path (default: print to stdout).")
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    if "employee_id" not in df.columns:
        raise SystemExit("Corpus must contain an employee_id column.")
    if "day" not in df.columns:
        df["day"] = df.groupby("employee_id").cumcount()  # fallback ordering

    if args.targets:
        targets = load_targets(Path(args.targets))
    else:
        print("WARNING: no --targets given; using self-referential neutral targets (smoke test only).")
        targets = neutral_targets_from_corpus(df)

    report = run_internal_validity(df, targets)

    print(f"Internal validity: {report['levels_passed']}/{report['total_levels']} levels, "
          f"overall {report['overall_score']:.3f}  (N={report['n_employees']} emp, {report['n_obs']} obs)")
    for lv in report["levels"]:
        print(f"  L{lv['level']} {lv['name']:<22s} {'PASS' if lv['passed'] else 'FAIL'}  score={lv['score']:.3f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
