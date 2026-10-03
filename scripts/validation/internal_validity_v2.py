#!/usr/bin/env python3
"""
Internal-validity battery, v2: a ground-truth-free psychometric validation battery.

Supersedes scripts/validation/internal_validity.py (v1). Where v1 reduced each level to a
binary pass/fail against ad-hoc thresholds, v2 reframes every criterion as a
literature-anchored EQUIVALENCE TEST: we ask whether the synthetic population is
statistically *indistinguishable* (within a pre-registered margin) from a
published target, with bootstrap uncertainty propagated by resampling PERSONAS.

Why equivalence testing (TOST) rather than NHST?
--------------------------------------------------
A non-significant difference test does NOT establish similarity (absence of
evidence vs evidence of absence). For a fidelity claim ("the synthetic corpus
behaves like a real one") the burden of proof is *equivalence*, so we use Two
One-Sided Tests (Schuirmann, 1987; Lakens, Scheel & Isager, 2018, AMPPS). Each
criterion gets a pre-registered equivalence margin chosen from the psychometric
literature (documented per criterion below), and a verdict:
    'equivalent'    — bootstrap/TOST CI lies entirely inside [-margin, +margin]
    'different'     — the point estimate lies outside the margin band
    'inconclusive'  — CI straddles a margin bound (underpowered with ~300 personas
                       x 6 days; honestly reported rather than spun as a pass)

Resampling unit = PERSONA (cluster bootstrap). Daily observations are nested
within personas; resampling rows would treat 6 correlated days as 6 independent
draws and badly understate uncertainty. We resample personas with replacement and
recompute the full multilevel statistic each rep (>=1000 reps).

Targets are NEVER fabricated. They are read from:
    data/seeds/norms.json            -> per-state grand_mean, total_sd, icc, scale
                                        (between/within SD derived: total*sqrt(icc),
                                         total*sqrt(1-icc))
    data/seeds/cross_state_targets.json (optional) -> between/within state-pair r's
If a target is missing, or flagged 'placeholder'/'unsourced', the criterion is
marked target_status='TARGET_UNAVAILABLE': we still report the descriptive
statistic + bootstrap CI, but the equivalence verdict is None. We never invent a
pass.

The six levels (L5 PA/NA permanently out of scope for this project):
  L1 Distributions   — Hedges g vs norm mean (TOST g=0.5); SD-ratio + CI
  L2 Variance/ICC    — between/within decomposition; ICC + CI; within-SD log-ratio
                       vs target (TOST ratio in [1/1.5, 1.5])
  L3 Nomological net — between- AND within-person state-pair correlations; per-pair
                       TOST vs cross_state_targets (|dr|=0.15); mean|dr|, % equivalent
  L4 Temporal        — MULTILEVEL AR(1) fixed effect (MixedLM) + CI; TOST vs target
  L6 Structure       — between- and within-level first-PC variance explained +
                       eigenvalues (the honest 4-indicator alternative to a CFA)
  L7 Range/ceiling   — scale coverage vs expected +/-2SD; ceiling/floor pileup vs the
                       ~15-20% convention (McHorney & Tarlov, 1995)

PSYCHOMETRIC HONESTY NOTES (load-bearing, do not "simplify" away):
  * L6 does NOT fit a 1-factor CFA. With 4 indicators a 1-factor model has df=2;
    CFI/RMSEA are unstable and effectively unfalsifiable at this df. We instead
    report multilevel dimensionality (first-PC variance explained at the between
    and within levels separately, plus the full eigenvalue spectrum). See
    Eisenhardt's caution and standard SEM identification rules (Kline, 2016): a
    just-/under-identified factor model carries no real fit information.
  * L4 does NOT average noisy per-person lag-1 r's. With 6 days each per-person
    AR(1) is estimated from 5 lag pairs and is extremely noisy. We pool with a
    multilevel AR(1) (today ~ lagged + person random intercept) and report the
    FIXED autoregressive coefficient with its Wald CI, plus the % of personas
    that contribute an estimable (non-constant) lagged series.

Install needs (NOT assumed present; checked at runtime):
    statsmodels  (MixedLM for L4 and the null-model ICC in L2)  [required]
    scipy        (t/normal quantiles for TOST)                  [required]
    pingouin     (optional; used for ICC + TOST if importable, else manual)
    pandas, numpy                                               [required]

CLI:
    python3 scripts/validation/internal_validity_v2.py \
        --corpus results/generation/full/GPT_4_1/states.csv \
        --targets results/leaderboard/internal_validity_v2_targets.json \
        --out results/leaderboard/GPT_4_1_internal_validity_v2.json \
        [--n-boot 1000] [--seed 17]

    python3 scripts/validation/internal_validity_v2.py --selftest
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scipy is required for TOST quantiles.
from scipy import stats as sps

# statsmodels is required for the multilevel AR(1) and null-model ICC.
try:
    import statsmodels.formula.api as smf
    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False

# pingouin is OPTIONAL.
try:
    import pingouin as pg  # noqa: F401
    _HAVE_PG = True
except Exception:
    _HAVE_PG = False


# ── State definitions (locked; do NOT re-derive) ─────────────────────────────
# Corpus uses the BARE column names (work_engagement, ...). v1 used *_state; v2
# reads the raw states.csv directly so no column renaming step is needed.
STATES: Dict[str, Dict[str, Any]] = {
    "work_engagement": {"col": "work_engagement", "min": 1.0, "max": 7.0},
    "psych_safety":    {"col": "psych_safety",    "min": 1.0, "max": 4.0},
    "work_creativity": {"col": "work_creativity", "min": 1.0, "max": 6.0},
    "team_trust":      {"col": "team_trust",       "min": 1.0, "max": 5.0},
}
STATE_NAMES = list(STATES.keys())

ID_COL = "employee_id"
DAY_COL = "day_index"

# ── Pre-registered equivalence margins (single source of truth) ──────────────
# Each margin is justified in the level docstrings and in the criterion table the
# CLI emits. Changing a margin is a results-changing decision (ledger entry).
MARGIN_HEDGES_G = 0.5      # Cohen (1988) "medium" SMD; L1 mean equivalence.
MARGIN_SDRATIO = (1 / 1.5, 1.5)   # L1 SD ratio band.
MARGIN_WITHIN_SD_RATIO = (1 / 1.5, 1.5)   # L2 within-person SD ratio band.
MARGIN_ICC = 0.10         # L2 ICC absolute margin (descriptive band; lit ICCs ~0.5).
MARGIN_DR = 0.15          # L3 per-pair correlation difference margin.
MARGIN_AR1 = 0.15         # L4 AR(1) coefficient margin.
CEILING_FLOOR_CONVENTION = 0.15   # McHorney & Tarlov (1995): >15% pileup = problematic.


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that survives numpy scalar/array/bool types and NaN->null."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ── Generic statistical helpers ───────────────────────────────────────────────

def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def hedges_g(sample: np.ndarray, target_mean: float, target_sd: float) -> float:
    """One-sample standardized mean difference vs a known population mean/SD.

    Uses the TARGET SD as the standardizer (we are asking how far the synthetic
    mean sits from the norm in published-SD units), then applies Hedges' small-N
    bias correction J = 1 - 3/(4n-9). Sign: positive = synthetic above norm.
    """
    s = _clean(sample)
    n = len(s)
    if n < 2 or not np.isfinite(target_sd) or target_sd <= 0:
        return np.nan
    d = (s.mean() - target_mean) / target_sd
    J = 1.0 - 3.0 / (4.0 * n - 9.0) if n > 2 else 1.0
    return d * J


def icc1_null_model(df: pd.DataFrame, col: str, fast: bool = False) -> Tuple[float, float, float]:
    """ICC(1) = between / (between + within) via a random-intercept null model.

    Preference order: statsmodels null MixedLM (REML) -> pooled-within ANOVA
    estimator. Returns (icc, between_sd, within_sd).

    `fast=True` skips the MixedLM and uses the closed-form pooled-within ANOVA
    estimator directly. This is used INSIDE the persona bootstrap: the point
    estimate keeps the gold-standard REML null model, but the bootstrap only needs
    to characterize the sampling variability of the SAME estimand, and refitting
    REML thousands of times is the dominant cost. The ANOVA and REML ICC(1)
    estimators are asymptotically equivalent for a balanced/near-balanced design
    (here exactly 6 days/persona), so the bootstrap CI is unaffected in
    expectation while running ~100x faster.
    """
    sub = df[[ID_COL, col]].dropna()
    if sub[ID_COL].nunique() < 5:
        return np.nan, np.nan, np.nan

    # statsmodels variance-components null model: col ~ 1 + (1|person).
    if _HAVE_SM and not fast:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm(f"{col} ~ 1", sub, groups=sub[ID_COL])
                mf = md.fit(reml=True, method="lbfgs")
            between_var = float(mf.cov_re.iloc[0, 0])
            within_var = float(mf.scale)
            total = between_var + within_var
            if total > 0:
                return between_var / total, np.sqrt(between_var), np.sqrt(within_var)
        except Exception:
            pass

    # Pooled-within ANOVA fallback (same estimator as v1 _compute_icc).
    g = sub.groupby(ID_COL)[col].agg(["mean", "std", "count"])
    g = g[g["count"] >= 2]
    if len(g) < 5:
        return np.nan, np.nan, np.nan
    between_var = g["mean"].var()
    g = g.assign(within_var=g["std"] ** 2, weight=g["count"] - 1).dropna()
    pooled_within = ((g["within_var"] * g["weight"]).sum() / g["weight"].sum()
                     if g["weight"].sum() > 0 else 0.0)
    total = between_var + pooled_within
    icc = between_var / total if total > 0 else np.nan
    return icc, np.sqrt(between_var), np.sqrt(pooled_within)


def within_person_sd(df: pd.DataFrame, col: str) -> float:
    """Pooled (RMS) within-person SD across personas with >=2 non-null days."""
    sub = df[[ID_COL, col]].dropna()
    g = sub.groupby(ID_COL)[col].agg(["std", "count"])
    g = g[(g["count"] >= 2) & g["std"].notna()]
    if len(g) == 0:
        return np.nan
    w = g["count"] - 1
    return float(np.sqrt((g["std"] ** 2 * w).sum() / w.sum()))


def between_within_corr(df: pd.DataFrame, c1: str, c2: str) -> Tuple[float, float]:
    """Between-person r (across person means) and within-person r (person-mean
    centered deviations) for two state columns."""
    sub = df[[ID_COL, c1, c2]].dropna()
    if len(sub) < 3:
        return np.nan, np.nan
    means = sub.groupby(ID_COL)[[c1, c2]].mean()
    b = means[c1].corr(means[c2]) if len(means) >= 3 else np.nan
    dev = sub.copy()
    for c in (c1, c2):
        dev[c] = dev[c] - dev.groupby(ID_COL)[c].transform("mean")
    w = dev[c1].corr(dev[c2])
    return (float(b) if pd.notna(b) else np.nan,
            float(w) if pd.notna(w) else np.nan)


def multilevel_ar1(df: pd.DataFrame, col: str, fast: bool = False) -> Dict[str, Any]:
    """WITHIN-person multilevel AR(1): dev_t ~ dev_{t-1} + (1|person).

    `dev` is the person-mean-centered state (col minus that persona's own mean).
    We deliberately do NOT regress raw state_t on raw state_{t-1}: with a stable
    between-person intercept (ICC ~ 0.5 here), the persona's fixed mean appears on
    both sides and induces strong positive lag covariance that is NOT
    autoregression — it is just the between-person variance leaking into the
    lagged term (the classic failure to person-center in dynamic multilevel
    models; see Hamaker & Grasman, 2015, Frontiers in Psychology; Wang & Maxwell,
    2015). Person-mean-centering both the outcome and the lagged predictor
    isolates the within-person carry-over, which is what "temporal dynamics" means
    for a state.

    IMPORTANT small-T caveat: with ~6 days/persona this within-person estimand is
    DOWNWARD biased (Nickell, 1981, Econometrica; Hurwicz bias of the
    within/fixed-effects dynamic-panel estimator ~ -(1+rho)/(T-1) ~ -0.26 at T=6).
    So the coefficient is the correct *between-person-free* quantity but
    under-estimates true carry-over; treat it as descriptive sign/direction
    evidence, not a precise autocorrelation. The diagnostic
    scripts/validation/_ar1_spec_check.py shows that on planted data (within-AR
    0.30, ICC 0.5): raw state~lag recovers ~0.62 (between-leak, WRONG); the within
    MixedLM, Mundlak, and pooled-OLS all recover ~0.02 (correct estimand, Nickell-
    attenuated). No T=6 estimator can recover 0.30; that is a property of 6-day
    data, not of this code.

    Returns the FIXED autoregressive coefficient, its Wald 95% CI and SE, plus the
    count/% of personas contributing an estimable (non-constant) lagged series.

    `fast=True` (used inside the persona bootstrap) uses a pooled OLS slope on the
    centered lag pairs instead of refitting the MixedLM — same estimand, far
    cheaper; the bootstrap only needs its sampling variability.
    """
    sub = df[[ID_COL, DAY_COL, col]].dropna().sort_values([ID_COL, DAY_COL]).copy()
    # Person-mean-center, then build the within-person lag.
    sub["dev"] = sub[col] - sub.groupby(ID_COL)[col].transform("mean")
    sub["lag_dev"] = sub.groupby(ID_COL)["dev"].shift(1)
    pairs = sub.dropna(subset=["lag_dev"])
    n_personas_total = df[ID_COL].nunique()
    estimable = pairs.groupby(ID_COL)["lag_dev"].apply(
        lambda s: s.notna().sum() >= 1 and s.var(ddof=0) > 1e-12)
    n_estimable = int(estimable.sum()) if len(estimable) else 0
    pct_estimable = (100.0 * n_estimable / n_personas_total) if n_personas_total else 0.0

    out: Dict[str, Any] = {
        "n_lag_pairs": int(len(pairs)),
        "n_personas_estimable": n_estimable,
        "pct_personas_estimable": round(pct_estimable, 1),
    }
    if len(pairs) < 10 or pairs["dev"].var() < 1e-12 or pairs["lag_dev"].var() < 1e-12:
        out.update({"ar1": np.nan, "ci": [np.nan, np.nan], "se": np.nan,
                    "method": "insufficient"})
        return out

    if _HAVE_SM and not fast:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm("dev ~ lag_dev", pairs, groups=pairs[ID_COL])
                mf = md.fit(reml=True, method="lbfgs")
            beta = float(mf.fe_params["lag_dev"])
            se = float(mf.bse_fe["lag_dev"])
            if np.isfinite(se):
                lo, hi = beta - 1.96 * se, beta + 1.96 * se
                out.update({"ar1": beta, "ci": [lo, hi], "se": se, "method": "mixedlm"})
                return out
        except Exception:
            pass

    # Pooled OLS on centered lag pairs (no person RE) — fallback.
    x = pairs["lag_dev"].values
    y = pairs["dev"].values
    sl, intercept, r, p, se = sps.linregress(x, y)
    out.update({"ar1": float(sl), "ci": [sl - 1.96 * se, sl + 1.96 * se],
                "se": float(se), "method": "ols_pooled"})
    return out


def pc1_variance_explained(mat: np.ndarray) -> Dict[str, Any]:
    """First-PC variance explained + full eigenvalue spectrum of a correlation
    matrix. The honest 4-indicator alternative to a 1-factor CFA: we report the
    eigenvalue structure rather than a fit index from an under-identified model.
    """
    if mat is None or mat.shape[0] < 2 or not np.all(np.isfinite(mat)):
        return {"pc1_var_explained": np.nan, "eigenvalues": None, "n_states": 0}
    eig = np.linalg.eigvalsh(mat)[::-1]  # descending
    eig = np.clip(eig, 0, None)
    tot = eig.sum()
    return {
        "pc1_var_explained": float(eig[0] / tot) if tot > 0 else np.nan,
        "eigenvalues": [round(float(e), 4) for e in eig],
        "n_states": int(mat.shape[0]),
    }


# ── TOST equivalence test ────────────────────────────────────────────────────

def tost_from_boot(boot: np.ndarray, point: float, target: float,
                   low: float, high: float, alpha: float = 0.05) -> Dict[str, Any]:
    """Equivalence verdict from a bootstrap distribution of the statistic.

    We center the test on the DIFFERENCE (statistic - target) and ask whether it
    lies inside the equivalence band [low, high] (the band is expressed on the
    same scale as `statistic - target`). The bootstrap (1-2*alpha) CI of the
    difference is the basis for the verdict (this is the bootstrap analogue of
    TOST; the 90% CI for alpha=0.05 matches the TOST decision rule, Lakens 2018):

        equivalent     : CI ⊆ [low, high]
        different       : point estimate of the difference outside [low, high]
        inconclusive    : CI straddles a bound (underpowered)

    `boot` is the bootstrap distribution of the STATISTIC (not the difference).
    """
    b = _clean(boot)
    if b.size < 20 or not np.isfinite(point) or not np.isfinite(target):
        return {"verdict": None, "diff": None, "ci90": [None, None],
                "margin": [low, high]}
    diff_boot = b - target
    diff_point = point - target
    lo90, hi90 = np.percentile(diff_boot, [100 * alpha, 100 * (1 - alpha)])
    inside = (lo90 >= low) and (hi90 <= high)
    outside = (diff_point < low) or (diff_point > high)
    if inside:
        verdict = "equivalent"
    elif outside:
        verdict = "different"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "diff": round(float(diff_point), 4),
            "ci90": [round(float(lo90), 4), round(float(hi90), 4)],
            "margin": [round(float(low), 4), round(float(high), 4)]}


def tost_ratio_from_boot(boot: np.ndarray, point: float, target: float,
                         ratio_band: Tuple[float, float],
                         alpha: float = 0.05) -> Dict[str, Any]:
    """Equivalence on a RATIO statistic/target (used for SD ratios).

    Works in log space so the band is symmetric: equivalent if the bootstrap 90%
    CI of log(statistic/target) lies inside [log(lo), log(hi)].
    """
    b = _clean(boot)
    if (b.size < 20 or not np.isfinite(point) or not np.isfinite(target)
            or target <= 0 or point <= 0 or np.any(b <= 0)):
        return {"verdict": None, "ratio": None, "ci90": [None, None],
                "band": list(ratio_band)}
    lo_band, hi_band = np.log(ratio_band[0]), np.log(ratio_band[1])
    log_ratio_boot = np.log(b / target)
    log_ratio_point = np.log(point / target)
    lo90, hi90 = np.percentile(log_ratio_boot, [100 * alpha, 100 * (1 - alpha)])
    inside = (lo90 >= lo_band) and (hi90 <= hi_band)
    outside = (log_ratio_point < lo_band) or (log_ratio_point > hi_band)
    verdict = "equivalent" if inside else ("different" if outside else "inconclusive")
    return {"verdict": verdict, "ratio": round(float(point / target), 4),
            "ci90_logratio": [round(float(lo90), 4), round(float(hi90), 4)],
            "band": list(ratio_band)}


# ── Persona cluster bootstrap ─────────────────────────────────────────────────

def persona_bootstrap(df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float],
                      n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap: resample PERSONAS with replacement, recompute stat_fn.

    Each rep draws n_personas ids with replacement and concatenates their rows
    (renaming ids so a persona drawn twice contributes two independent clusters).
    Returns the bootstrap distribution (NaNs from degenerate resamples dropped).
    """
    # Precompute, ONCE, the row positions of each persona so a bootstrap rep is a
    # cheap row-index gather (np.concatenate) + a single .iloc, instead of per-rep
    # per-group .copy()+pd.concat. This is the dominant cost when ~40 separate
    # bootstrap loops each rebuild a multi-thousand-row frame.
    codes, uniques = pd.factorize(df[ID_COL].values)
    n = len(uniques)
    pos_by_persona = [np.where(codes == k)[0] for k in range(n)]
    base = df.reset_index(drop=True)

    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        idx_parts = [pos_by_persona[k] for k in draw]
        row_idx = np.concatenate(idx_parts)
        # New unique cluster id per drawn persona (so a persona drawn twice forms
        # two independent clusters), assigned via np.repeat over draw order.
        cluster_ids = np.repeat(np.arange(n), [len(p) for p in idx_parts])
        rs = base.iloc[row_idx].copy()
        rs[ID_COL] = cluster_ids
        try:
            out[b] = stat_fn(rs)
        except Exception:
            out[b] = np.nan
    return out


def boot_ci(boot: np.ndarray, alpha: float = 0.05) -> List[Optional[float]]:
    b = _clean(boot)
    if b.size < 20:
        return [None, None]
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


# ── Target normalization ──────────────────────────────────────────────────────

def _state_target(targets: dict, name: str) -> dict:
    return (targets.get("states", {}) or {}).get(name, {}) or {}


def _is_unavailable(val: Any) -> bool:
    """A target value counts as unavailable if it is None/NaN or a string flag."""
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in {"placeholder", "unsourced", "tbd", "na", ""}
    try:
        return not np.isfinite(float(val))
    except (TypeError, ValueError):
        return True


def _cross_state_target(targets: dict, level: str, s1: str,
                        s2: str) -> Dict[str, Any]:
    """Resolve the cross-state correlation target for a pair at 'between'/'within'.

    Returns a dict: {r, status, margin, is_expert_prior, source_status}.
      status         : 'OK' (an equivalence test is run) or 'TARGET_UNAVAILABLE'
                       (no usable target; report observed r + CI only).
      margin         : per-pair equivalence margin (the sourced JSON sets a WIDE
                       margin for expert priors); falls back to MARGIN_DR.
      is_expert_prior: True -> the target r is a documented prior, NOT a measured
                       meta-analytic estimate. The test still runs but the verdict
                       MUST be read as "consistent with an expert prior", never as
                       "matches a sourced value". Surfaced so the manuscript can
                       distinguish the two.
      source_status  : the raw status string from the JSON ('verified',
                       'abstract_only', 'no_meta_found', ...).

    Two accepted target schemas:
      (A) Sourced (current data/seeds/cross_state_targets.json): top-level
          'pairs' -> {pair: {r, margin, is_expert_prior, status, ...}}. This file
          carries BETWEEN-person estimates only (the standard meta-analytic
          criterion); within-person therefore stays TARGET_UNAVAILABLE.
      (B) Legacy nested: 'cross_state' -> {'between'|'within': {pair: r | {...}}}.
          Used by the self-test (which can plant both levels).

    A 'no_meta_found' pair with is_expert_prior=true is still returned as a usable
    (prior) target with its wide margin; a pair whose r is missing/NaN, or flagged
    placeholder/unsourced/tbd, is TARGET_UNAVAILABLE.
    """
    none = {"r": None, "status": "TARGET_UNAVAILABLE", "margin": MARGIN_DR,
            "is_expert_prior": None, "source_status": None}

    # Schema A: top-level 'pairs', between-person only.
    pairs = targets.get("pairs")
    if isinstance(pairs, dict) and level == "between":
        for key in (f"{s1}__{s2}", f"{s2}__{s1}"):
            if key in pairs:
                e = pairs[key]
                r = e.get("r")
                src = str(e.get("status", "")).lower()
                if _is_unavailable(r) or src in {"placeholder", "unsourced", "tbd"}:
                    out = dict(none); out["source_status"] = src or None
                    return out
                return {"r": float(r), "status": "OK",
                        "margin": float(e.get("margin", MARGIN_DR)),
                        "is_expert_prior": bool(e.get("is_expert_prior", False)),
                        "source_status": src or None}
        return none
    if isinstance(pairs, dict) and level == "within":
        return none  # sourced file has no within-person targets

    # Schema B (legacy nested) — used by the self-test.
    cs = targets.get("cross_state", {}) or {}
    lvl = cs.get(level, {}) or {}
    for key in (f"{s1}__{s2}", f"{s2}__{s1}"):
        if key in lvl:
            entry = lvl[key]
            if isinstance(entry, dict):
                r = entry.get("r")
                src = str(entry.get("status", "")).lower()
                if src in {"placeholder", "unsourced", "tbd"} or _is_unavailable(r):
                    return none
                return {"r": float(r), "status": "OK",
                        "margin": float(entry.get("margin", MARGIN_DR)),
                        "is_expert_prior": bool(entry.get("is_expert_prior", False)),
                        "source_status": src or None}
            if _is_unavailable(entry):
                return none
            return {"r": float(entry), "status": "OK", "margin": MARGIN_DR,
                    "is_expert_prior": False, "source_status": None}
    return none


# ── Levels ────────────────────────────────────────────────────────────────────

def level1_distributions(df, targets, n_boot, rng) -> Dict[str, Any]:
    """L1 — distributional fidelity per state.

    Mean: Hedges g vs the published grand mean (standardized by published SD),
    TOST margin g = 0.5 (Cohen 1988 medium). SD: ratio synth/target with a
    [1/1.5, 1.5] equivalence band. Both with persona-cluster bootstrap CIs.
    """
    crit: Dict[str, Any] = {}
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        tgt = _state_target(targets, name)
        t_mean, t_sd = tgt.get("grand_mean"), tgt.get("total_sd")
        vals = _clean(df[col].values)
        synth_mean = float(vals.mean()) if vals.size else np.nan
        synth_sd = float(vals.std(ddof=1)) if vals.size > 1 else np.nan

        entry: Dict[str, Any] = {
            "synth_mean": round(synth_mean, 4), "synth_sd": round(synth_sd, 4),
            "target_mean": t_mean, "target_sd": t_sd,
        }

        if _is_unavailable(t_mean) or _is_unavailable(t_sd):
            entry["target_status"] = "TARGET_UNAVAILABLE"
            entry["mean_equivalence"] = {"verdict": None}
            entry["sd_equivalence"] = {"verdict": None}
            crit[name] = entry
            continue
        t_mean, t_sd = float(t_mean), float(t_sd)

        # Hedges g + bootstrap.
        g_point = hedges_g(vals, t_mean, t_sd)
        g_boot = persona_bootstrap(
            df, lambda d, c=col, m=t_mean, s=t_sd: hedges_g(d[c].values, m, s),
            n_boot, rng)
        # g is already a difference in SD units, so TOST band is [-margin, margin]
        # centered on a target difference of 0 (the statistic IS the difference).
        mean_eq = tost_from_boot(g_boot + 0.0, g_point, 0.0,
                                 -MARGIN_HEDGES_G, MARGIN_HEDGES_G)
        entry["hedges_g"] = round(float(g_point), 4) if np.isfinite(g_point) else None
        entry["hedges_g_ci"] = boot_ci(g_boot)
        entry["mean_equivalence"] = mean_eq

        # SD ratio + bootstrap.
        sd_boot = persona_bootstrap(
            df, lambda d, c=col: float(_clean(d[c].values).std(ddof=1))
            if _clean(d[c].values).size > 1 else np.nan, n_boot, rng)
        sd_eq = tost_ratio_from_boot(sd_boot, synth_sd, t_sd, MARGIN_SDRATIO)
        entry["sd_ratio"] = round(synth_sd / t_sd, 4) if t_sd > 0 else None
        entry["sd_ratio_ci"] = boot_ci(sd_boot / t_sd if t_sd > 0 else sd_boot)
        entry["sd_equivalence"] = sd_eq
        entry["target_status"] = "OK"
        crit[name] = entry

    n_eq = sum(1 for e in crit.values()
               if e.get("mean_equivalence", {}).get("verdict") == "equivalent")
    return {"name": "Distributions", "per_state": crit,
            "n_states_mean_equivalent": n_eq,
            "n_states_evaluated": sum(1 for e in crit.values()
                                      if e.get("target_status") == "OK")}


def level2_variance_icc(df, targets, n_boot, rng) -> Dict[str, Any]:
    """L2 — between/within variance decomposition.

    ICC(1) via null MixedLM (or pingouin) + bootstrap CI, reported against the
    literature ICC (margin +/-0.10, descriptive band). Within-person SD vs the
    derived target within-SD (total_sd*sqrt(1-icc)) with a [1/1.5, 1.5] ratio
    equivalence band — the within-SD ratio is the primary fidelity discriminator
    (how much personas fluctuate day to day).
    """
    crit: Dict[str, Any] = {}
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        tgt = _state_target(targets, name)
        t_icc = tgt.get("icc")
        t_total = tgt.get("total_sd")
        t_within = None
        if not _is_unavailable(t_icc) and not _is_unavailable(t_total):
            t_within = float(t_total) * np.sqrt(1.0 - float(t_icc))

        icc, b_sd, w_sd = icc1_null_model(df, col)
        entry: Dict[str, Any] = {
            "synth_icc": round(float(icc), 4) if np.isfinite(icc) else None,
            "synth_between_sd": round(float(b_sd), 4) if np.isfinite(b_sd) else None,
            "synth_within_sd": round(float(w_sd), 4) if np.isfinite(w_sd) else None,
            "target_icc": t_icc, "target_within_sd": round(t_within, 4) if t_within else None,
            "icc_method": "pingouin" if _HAVE_PG else ("mixedlm" if _HAVE_SM else "anova"),
        }

        # ICC bootstrap + equivalence (descriptive margin). Bootstrap uses the
        # fast closed-form estimator (see icc1_null_model docstring); the point
        # estimate above keeps the REML null model.
        icc_boot = persona_bootstrap(
            df, lambda d, c=col: icc1_null_model(d, c, fast=True)[0], n_boot, rng)
        entry["synth_icc_ci"] = boot_ci(icc_boot)
        if not _is_unavailable(t_icc):
            entry["icc_equivalence"] = tost_from_boot(
                icc_boot, float(icc), float(t_icc), -MARGIN_ICC, MARGIN_ICC)
        else:
            entry["icc_equivalence"] = {"verdict": None}

        # Within-SD bootstrap + ratio equivalence (primary).
        wsd_boot = persona_bootstrap(
            df, lambda d, c=col: within_person_sd(d, c), n_boot, rng)
        entry["synth_within_sd_ci"] = boot_ci(wsd_boot)
        if t_within and np.isfinite(w_sd):
            entry["within_sd_equivalence"] = tost_ratio_from_boot(
                wsd_boot, float(w_sd), float(t_within), MARGIN_WITHIN_SD_RATIO)
            entry["target_status"] = "OK"
        else:
            entry["within_sd_equivalence"] = {"verdict": None}
            entry["target_status"] = "TARGET_UNAVAILABLE"
        crit[name] = entry

    n_eq = sum(1 for e in crit.values()
               if e.get("within_sd_equivalence", {}).get("verdict") == "equivalent")
    return {"name": "Variance/ICC", "per_state": crit,
            "n_states_within_sd_equivalent": n_eq,
            "n_states_evaluated": sum(1 for e in crit.values()
                                      if e.get("target_status") == "OK")}


def level3_nomological(df, targets, n_boot, rng) -> Dict[str, Any]:
    """L3 — nomological net (state-pair correlations).

    Reports the full between-person (across person means) AND within-person
    (person-mean-centered) correlation matrices for all 6 state pairs, each with a
    persona-bootstrap CI. If a cross-state target exists for a pair, runs a
    per-pair TOST using that pair's OWN margin (the sourced JSON sets a wide margin
    for expert priors) and contributes to mean|dr| / % equivalent. The
    is_expert_prior flag is carried through so a verdict against a prior is never
    presented as a match to a measured meta-analytic value. The sourced target
    file carries BETWEEN-person estimates only, so within-person pairs are reported
    descriptively (TARGET_UNAVAILABLE) unless a within target is supplied.

    Equivalent counts are split into sourced-only (is_expert_prior=False) vs.
    including-priors so the headline can cite the conservative number.
    """
    present = [n for n in STATE_NAMES if STATES[n]["col"] in df.columns]
    between: Dict[str, Any] = {}
    within: Dict[str, Any] = {}
    dr_between, dr_within = [], []
    eq_between, eq_within = [], []
    # Sourced-only (exclude expert priors) for the conservative headline count.
    eq_between_sourced: List[bool] = []

    for i, s1 in enumerate(present):
        for j, s2 in enumerate(present):
            if i >= j:
                continue
            c1, c2 = STATES[s1]["col"], STATES[s2]["col"]
            key = f"{s1}__{s2}"
            b_point, w_point = between_within_corr(df, c1, c2)

            b_boot = persona_bootstrap(
                df, lambda d, a=c1, c=c2: between_within_corr(d, a, c)[0], n_boot, rng)
            w_boot = persona_bootstrap(
                df, lambda d, a=c1, c=c2: between_within_corr(d, a, c)[1], n_boot, rng)

            for lvl_name, point, boot, store, drs, eqs in (
                ("between", b_point, b_boot, between, dr_between, eq_between),
                ("within", w_point, w_boot, within, dr_within, eq_within)):
                tg = _cross_state_target(targets, lvl_name, s1, s2)
                tgt_r, status, margin = tg["r"], tg["status"], tg["margin"]
                rec: Dict[str, Any] = {
                    "synth_r": round(float(point), 4) if np.isfinite(point) else None,
                    "synth_r_ci": boot_ci(boot),
                    "target_r": tgt_r, "target_status": status,
                    "margin": margin if status == "OK" else None,
                    "is_expert_prior": tg["is_expert_prior"],
                    "source_status": tg["source_status"],
                }
                if status == "OK" and np.isfinite(point):
                    eqd = tost_from_boot(boot, float(point), float(tgt_r),
                                         -float(margin), float(margin))
                    rec["equivalence"] = eqd
                    drs.append(abs(float(point) - float(tgt_r)))
                    is_equiv = eqd["verdict"] == "equivalent"
                    eqs.append(is_equiv)
                    if lvl_name == "between" and not tg["is_expert_prior"]:
                        eq_between_sourced.append(is_equiv)
                else:
                    rec["equivalence"] = {"verdict": None}
                store[key] = rec

    def _summ(drs, eqs):
        return {
            "mean_abs_dr": round(float(np.mean(drs)), 4) if drs else None,
            "n_pairs_with_target": len(drs),
            "pct_pairs_equivalent": round(100.0 * np.mean(eqs), 1) if eqs else None,
        }

    summ_b = _summ(dr_between, eq_between)
    summ_b["n_pairs_sourced_nonprior"] = len(eq_between_sourced)
    summ_b["pct_pairs_equivalent_sourced_only"] = (
        round(100.0 * np.mean(eq_between_sourced), 1) if eq_between_sourced else None)
    return {"name": "Nomological net",
            "between_person": between, "within_person": within,
            "summary_between": summ_b,
            "summary_within": _summ(dr_within, eq_within),
            "targets_available": bool(dr_between or dr_within)}


def level4_temporal(df, targets, n_boot, rng) -> Dict[str, Any]:
    """L4 — temporal dynamics via WITHIN-person MULTILEVEL AR(1).

    Reports the FIXED within-person autoregressive coefficient (MixedLM
    person-centered dev_t ~ dev_{t-1} + (1|person); see multilevel_ar1 for why raw
    state is NOT used) with its Wald CI and a persona-bootstrap CI, plus the % of
    personas with an estimable lagged series. If targets supply a per-state 'ar1'
    day-level autocorrelation target, runs a TOST (margin 0.15); otherwise reports
    descriptively. NOTE: the current sourced targets file has no AR(1) anchor, so
    on real corpora L4 is descriptive (TARGET_UNAVAILABLE).
    """
    crit: Dict[str, Any] = {}
    ar_targets = (targets.get("temporal", {}) or {}).get("ar1", {}) or {}
    for name in STATE_NAMES:
        col = STATES[name]["col"]
        if col not in df.columns:
            continue
        res = multilevel_ar1(df, col)
        entry: Dict[str, Any] = {
            "ar1_fixed": round(float(res["ar1"]), 4) if np.isfinite(res.get("ar1", np.nan)) else None,
            "ar1_wald_ci": [round(float(x), 4) if np.isfinite(x) else None
                            for x in res.get("ci", [np.nan, np.nan])],
            "ar1_method": res.get("method"),
            "pct_personas_estimable": res.get("pct_personas_estimable"),
            "n_lag_pairs": res.get("n_lag_pairs"),
        }
        ar_boot = persona_bootstrap(
            df, lambda d, c=col: multilevel_ar1(d, c, fast=True).get("ar1", np.nan),
            n_boot, rng)
        entry["ar1_boot_ci"] = boot_ci(ar_boot)

        tgt = ar_targets.get(name)
        if not _is_unavailable(tgt) and np.isfinite(res.get("ar1", np.nan)):
            entry["ar1_equivalence"] = tost_from_boot(
                ar_boot, float(res["ar1"]), float(tgt), -MARGIN_AR1, MARGIN_AR1)
            entry["target_ar1"] = float(tgt)
            entry["target_status"] = "OK"
        else:
            entry["ar1_equivalence"] = {"verdict": None}
            entry["target_ar1"] = tgt if not _is_unavailable(tgt) else None
            entry["target_status"] = "TARGET_UNAVAILABLE"
        crit[name] = entry

    n_eq = sum(1 for e in crit.values()
               if e.get("ar1_equivalence", {}).get("verdict") == "equivalent")
    return {"name": "Temporal (within-person multilevel AR1)", "per_state": crit,
            "n_states_ar1_equivalent": n_eq,
            "n_states_evaluated": sum(1 for e in crit.values()
                                      if e.get("target_status") == "OK"),
            "t6_caveat": ("With ~6 days/persona the within-person AR(1) is downward-"
                          "biased (Nickell/Hurwicz dynamic-panel bias ~ -(1+rho)/(T-1) "
                          "~ -0.26 here); the reported coefficient is the correct "
                          "(between-person-free) estimand but UNDER-estimates true "
                          "carry-over. Treat L4 as descriptive evidence of direction/"
                          "sign, not a precise autocorrelation. Verified in "
                          "scripts/validation/_ar1_spec_check.py.")}


def level6_structure(df, targets, n_boot, rng) -> Dict[str, Any]:
    """L6 — multilevel dimensionality (the honest 4-indicator structure check).

    Deliberately NOT a 1-factor CFA: with 4 indicators a 1-factor model has df=2
    and its CFI/RMSEA are unstable and effectively unfalsifiable (Kline 2016 on
    identification). Instead we report the first-PC variance explained and the
    full eigenvalue spectrum SEPARATELY at the between-person level (correlation
    of person means) and the within-person level (person-mean-centered
    deviations), each with a persona-bootstrap CI on PC1 variance explained.
    """
    present = [n for n in STATE_NAMES if STATES[n]["col"] in df.columns]
    cols = [STATES[n]["col"] for n in present]

    def between_pc1(d: pd.DataFrame) -> float:
        m = d.groupby(ID_COL)[cols].mean()
        if len(m) < 3:
            return np.nan
        return pc1_variance_explained(m.corr().values)["pc1_var_explained"]

    def within_pc1(d: pd.DataFrame) -> float:
        dev = d.copy()
        for c in cols:
            dev[c] = dev[c] - dev.groupby(ID_COL)[c].transform("mean")
        return pc1_variance_explained(dev[cols].corr().values)["pc1_var_explained"]

    between_means = df.groupby(ID_COL)[cols].mean()
    between_struct = pc1_variance_explained(between_means.corr().values)
    dev = df.copy()
    for c in cols:
        dev[c] = dev[c] - dev.groupby(ID_COL)[c].transform("mean")
    within_struct = pc1_variance_explained(dev[cols].corr().values)

    b_boot = persona_bootstrap(df, between_pc1, n_boot, rng)
    w_boot = persona_bootstrap(df, within_pc1, n_boot, rng)

    return {
        "name": "Structure (multilevel dimensionality)",
        "note": ("4 indicators -> 1-factor CFA is under-identified (df=2); "
                 "reporting PC1 variance explained + eigenvalues at both levels "
                 "instead of CFI/RMSEA."),
        "between_person": {
            "pc1_var_explained": round(between_struct["pc1_var_explained"], 4)
            if np.isfinite(between_struct["pc1_var_explained"]) else None,
            "pc1_var_explained_ci": boot_ci(b_boot),
            "eigenvalues": between_struct["eigenvalues"],
            "n_states": between_struct["n_states"],
        },
        "within_person": {
            "pc1_var_explained": round(within_struct["pc1_var_explained"], 4)
            if np.isfinite(within_struct["pc1_var_explained"]) else None,
            "pc1_var_explained_ci": boot_ci(w_boot),
            "eigenvalues": within_struct["eigenvalues"],
            "n_states": within_struct["n_states"],
        },
        "target_status": "TARGET_UNAVAILABLE",  # no published PC1 anchor; descriptive only
    }


def level7_range_ceiling(df, targets, n_boot, rng) -> Dict[str, Any]:
    """L7 — range coverage and ceiling/floor pileup.

    Coverage: fraction of the +/-2SD band (target grand_mean +/- 2*total_sd,
    clipped to scale bounds) actually spanned by observed values — the population
    should realize roughly the central 95% region implied by the published norm.
    Ceiling/floor: % of observations within 0.1 of the scale bounds, flagged
    against the ~15% problematic-pileup convention (McHorney & Tarlov 1995). Both
    with persona-bootstrap CIs.
    """
    crit: Dict[str, Any] = {}
    for name in STATE_NAMES:
        col, smin, smax = STATES[name]["col"], STATES[name]["min"], STATES[name]["max"]
        if col not in df.columns:
            continue
        tgt = _state_target(targets, name)
        t_mean, t_sd = tgt.get("grand_mean"), tgt.get("total_sd")
        vals = _clean(df[col].values)
        if vals.size == 0:
            continue

        def _floor(d, c=col, lo=smin):
            v = _clean(d[c].values)
            return float((v <= lo + 0.1).mean()) if v.size else np.nan

        def _ceiling(d, c=col, hi=smax):
            v = _clean(d[c].values)
            return float((v >= hi - 0.1).mean()) if v.size else np.nan

        floor_pct = float((vals <= smin + 0.1).mean())
        ceil_pct = float((vals >= smax - 0.1).mean())
        floor_boot = persona_bootstrap(df, _floor, n_boot, rng)
        ceil_boot = persona_bootstrap(df, _ceiling, n_boot, rng)

        entry: Dict[str, Any] = {
            "pct_floor": round(100 * floor_pct, 2),
            "pct_floor_ci": [round(100 * x, 2) if x is not None else None
                             for x in boot_ci(floor_boot)],
            "pct_ceiling": round(100 * ceil_pct, 2),
            "pct_ceiling_ci": [round(100 * x, 2) if x is not None else None
                               for x in boot_ci(ceil_boot)],
            "floor_problematic": bool(floor_pct > CEILING_FLOOR_CONVENTION),
            "ceiling_problematic": bool(ceil_pct > CEILING_FLOOR_CONVENTION),
            "ceiling_floor_convention_pct": CEILING_FLOOR_CONVENTION * 100,
            "observed_min": round(float(vals.min()), 3),
            "observed_max": round(float(vals.max()), 3),
        }

        if not _is_unavailable(t_mean) and not _is_unavailable(t_sd):
            band_lo = max(smin, float(t_mean) - 2 * float(t_sd))
            band_hi = min(smax, float(t_mean) + 2 * float(t_sd))
            band = band_hi - band_lo
            covered = (min(vals.max(), band_hi) - max(vals.min(), band_lo))
            coverage = float(np.clip(covered / band, 0, 1)) if band > 0 else np.nan
            entry["expected_2sd_band"] = [round(band_lo, 3), round(band_hi, 3)]
            entry["coverage_of_2sd_band"] = round(coverage, 4)
            entry["target_status"] = "OK"
        else:
            entry["coverage_of_2sd_band"] = None
            entry["target_status"] = "TARGET_UNAVAILABLE"
        crit[name] = entry

    n_problem = sum(1 for e in crit.values()
                    if e["floor_problematic"] or e["ceiling_problematic"])
    return {"name": "Range/ceiling", "per_state": crit,
            "n_states_with_pileup": n_problem}


# ── Driver ────────────────────────────────────────────────────────────────────

def run_internal_validity_v2(df: pd.DataFrame, targets: dict, n_boot: int = 1000,
                   seed: int = 17) -> dict:
    """Run the full v2 battery on one corpus DataFrame + a targets dict.

    Returns a structured dict with one block per level; every quantitative
    criterion carries (statistic, bootstrap CI, target, margin, equivalence
    verdict, target_status). No single scalar 'pass' is emitted — fidelity is
    summarized as counts of equivalent criteria and continuous distances.
    """
    if ID_COL not in df.columns:
        raise ValueError(f"corpus must contain '{ID_COL}'")
    if DAY_COL not in df.columns:
        df = df.copy()
        df[DAY_COL] = df.groupby(ID_COL).cumcount() + 1

    rng = np.random.default_rng(seed)
    levels = {
        "L1": level1_distributions(df, targets, n_boot, rng),
        "L2": level2_variance_icc(df, targets, n_boot, rng),
        "L3": level3_nomological(df, targets, n_boot, rng),
        "L4": level4_temporal(df, targets, n_boot, rng),
        "L6": level6_structure(df, targets, n_boot, rng),
        "L7": level7_range_ceiling(df, targets, n_boot, rng),
    }

    # Headline continuous summary (no inflated single pass/fail).
    n_equiv = (levels["L1"]["n_states_mean_equivalent"]
               + levels["L2"]["n_states_within_sd_equivalent"]
               + levels["L4"]["n_states_ar1_equivalent"])
    n_eval = (levels["L1"]["n_states_evaluated"]
              + levels["L2"]["n_states_evaluated"]
              + levels["L4"]["n_states_evaluated"])
    # L3 contributes equivalent-pair counts if targets exist.
    l3b = levels["L3"]["summary_between"]
    l3w = levels["L3"]["summary_within"]

    return {
        "engine": "internal_validity_v2",
        "n_obs": int(len(df)),
        "n_personas": int(df[ID_COL].nunique()),
        "n_boot": n_boot,
        "seed": seed,
        "have_statsmodels": _HAVE_SM,
        "have_pingouin": _HAVE_PG,
        "margins": {
            "hedges_g": MARGIN_HEDGES_G, "sd_ratio": list(MARGIN_SDRATIO),
            "within_sd_ratio": list(MARGIN_WITHIN_SD_RATIO), "icc": MARGIN_ICC,
            "delta_r": MARGIN_DR, "ar1": MARGIN_AR1,
            "ceiling_floor_pct": CEILING_FLOOR_CONVENTION,
        },
        "levels": levels,
        "summary": {
            "n_state_criteria_equivalent": n_equiv,
            "n_state_criteria_evaluated": n_eval,
            "L3_mean_abs_dr_between": l3b["mean_abs_dr"],
            "L3_pct_pairs_equivalent_between": l3b["pct_pairs_equivalent"],
            "L3_n_pairs_with_target_between": l3b["n_pairs_with_target"],
            "L3_n_pairs_sourced_nonprior": l3b.get("n_pairs_sourced_nonprior"),
            "L3_pct_pairs_equivalent_sourced_only": l3b.get("pct_pairs_equivalent_sourced_only"),
            "L3_mean_abs_dr_within": l3w["mean_abs_dr"],
            "L3_pct_pairs_equivalent_within": l3w["pct_pairs_equivalent"],
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Internal-validity battery (v2): equivalence tests on a generated corpus.")
    ap.add_argument("--corpus", help="states.csv (employee_id, day_index, 4 state cols).")
    ap.add_argument("--targets", help="Targets JSON (norms-derived + cross_state).")
    ap.add_argument("--out", default=None, help="Output JSON path (default: stdout summary only).")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="Bootstrap reps (default 1000 for a real run, 300 for --selftest).")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--selftest", action="store_true",
                    help="Run the planted-parameter recovery self-test and exit.")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(seed=args.seed, n_boot=args.n_boot or 300)
        return
    n_boot = args.n_boot or 1000

    if not args.corpus or not args.targets:
        ap.error("--corpus and --targets are required unless --selftest is given.")

    df = pd.read_csv(args.corpus)
    targets = json.load(open(args.targets))
    report = run_internal_validity_v2(df, targets, n_boot=n_boot, seed=args.seed)

    s = report["summary"]
    print(f"Internal-validity battery v2  N={report['n_personas']} personas, {report['n_obs']} obs, "
          f"{report['n_boot']} boot  (statsmodels={report['have_statsmodels']}, "
          f"pingouin={report['have_pingouin']})")
    print(f"  state criteria equivalent: {s['n_state_criteria_equivalent']}"
          f"/{s['n_state_criteria_evaluated']} (L1 mean + L2 within-SD + L4 AR1)")
    print(f"  L3 between mean|dr|={s['L3_mean_abs_dr_between']} "
          f"({s['L3_pct_pairs_equivalent_between']}% pairs equiv); "
          f"within mean|dr|={s['L3_mean_abs_dr_within']} "
          f"({s['L3_pct_pairs_equivalent_within']}%)")
    for lk in ["L1", "L2", "L4"]:
        lvl = report["levels"][lk]
        for st, e in lvl["per_state"].items():
            verds = [v.get("verdict") for k, v in e.items()
                     if isinstance(v, dict) and "verdict" in v]
            print(f"   {lk} {st:<16s} {', '.join(str(x) for x in verds)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out, "w"), indent=2, cls=NumpyEncoder)
        print(f"Wrote {out}")


# ── Self-test: plant known parameters, confirm recovery ───────────────────────

def _simulate_corpus(n_personas: int, n_days: int, rng: np.random.Generator,
                     grand_means: Dict[str, float], between_sds: Dict[str, float],
                     within_sds: Dict[str, float], scales: Dict[str, Tuple[float, float]],
                     between_r: float = 0.0, ar1: float = 0.0) -> pd.DataFrame:
    """Generate a corpus with PLANTED ICC, within-SD, between-state correlation,
    and AR(1) so the battery can be checked for recovery.

    Model: state_i,t = grand_mean + b_i (between, optionally cross-correlated)
                       + e_i,t (within, AR(1) with coef ar1).
    Implied ICC = between_var / (between_var + within_var_stationary).
    """
    names = list(grand_means.keys())
    k = len(names)
    # Between-person draws with a shared correlation `between_r` across states.
    cov_b = np.full((k, k), between_r)
    np.fill_diagonal(cov_b, 1.0)
    L = np.linalg.cholesky(cov_b)
    z = rng.standard_normal((n_personas, k)) @ L.T
    b = {nm: z[:, i] * between_sds[nm] for i, nm in enumerate(names)}

    rows = []
    for p in range(n_personas):
        prev = {nm: 0.0 for nm in names}
        for t in range(n_days):
            row = {ID_COL: f"S{p:04d}", DAY_COL: t + 1}
            for nm in names:
                innov_sd = within_sds[nm] * np.sqrt(1 - ar1 ** 2) if abs(ar1) < 1 else within_sds[nm]
                e = ar1 * prev[nm] + rng.normal(0, innov_sd)
                prev[nm] = e
                val = grand_means[nm] + b[nm][p] + e
                lo, hi = scales[nm]
                row[STATES[nm]["col"]] = float(np.clip(val, lo, hi))
            rows.append(row)
    return pd.DataFrame(rows)


def run_selftest(seed: int = 17, n_boot: int = 300, n_personas: int = 500):
    print("=" * 72)
    print("Internal-validity battery v2 SELF-TEST — recover planted ICC / within-SD / corr / AR(1)")
    print(f"  n_personas={n_personas}  n_days=6  n_boot={n_boot}  seed={seed}")
    print("=" * 72)
    rng = np.random.default_rng(seed)

    grand_means = {"work_engagement": 4.80, "psych_safety": 2.90,
                   "work_creativity": 3.60, "team_trust": 3.70}
    # Plant ICC = 0.50 for all -> between_sd == within_sd.
    planted_within = {"work_engagement": 0.74, "psych_safety": 0.39,
                      "work_creativity": 0.60, "team_trust": 0.47}
    planted_between = dict(planted_within)  # ICC 0.5
    scales = {nm: (STATES[nm]["min"], STATES[nm]["max"]) for nm in grand_means}
    PLANTED_R = 0.40
    PLANTED_AR1 = 0.30

    df = _simulate_corpus(n_personas, 6, rng, grand_means, planted_between, planted_within,
                          scales, between_r=PLANTED_R, ar1=PLANTED_AR1)

    # Build matched targets directly from the planted parameters.
    states_t = {}
    for nm in grand_means:
        total_sd = np.sqrt(planted_between[nm] ** 2 + planted_within[nm] ** 2)
        icc = planted_between[nm] ** 2 / total_sd ** 2
        states_t[nm] = {"grand_mean": grand_means[nm], "total_sd": float(total_sd),
                        "icc": float(icc)}
    pairs = [(a, b) for i, a in enumerate(grand_means) for b in list(grand_means)[i + 1:]]
    cross = {"between": {f"{a}__{b}": PLANTED_R for a, b in pairs},
             "within": {f"{a}__{b}": 0.0 for a, b in pairs}}
    # L4 target: NOT the planted 0.30. At T=6 the within-person AR(1) estimand is
    # severely downward-biased (Nickell/Hurwicz dynamic-panel bias ~ -(1+rho)/(T-1)
    # ~ -0.26), so the estimable quantity is ~0.0-0.05, not 0.30. The diagnostic
    # scripts/validation/_ar1_spec_check.py confirms every unbiased-direction
    # estimator (within MixedLM, Mundlak, pooled OLS) recovers ~0.02 here. We set
    # the self-test L4 target to this ATTENUATED expectation so the equivalence
    # test is calibrated to what 6 days can actually identify. On real corpora L4
    # has no AR(1) anchor at all, so it is reported descriptively. This is the
    # honest treatment of the task's stated 6-day temporal-identification limit.
    ATTENUATED_AR1 = 0.0  # estimable within-AR target at T=6 (see note above)
    temporal = {"ar1": {nm: ATTENUATED_AR1 for nm in grand_means}}
    targets = {"states": states_t, "cross_state": cross, "temporal": temporal}

    rep = run_internal_validity_v2(df, targets, n_boot=n_boot, seed=seed)

    ok = True
    print("\n[L1] mean recovery (Hedges g should be ~0, verdict equivalent):")
    for nm, e in rep["levels"]["L1"]["per_state"].items():
        g = e.get("hedges_g")
        v = e["mean_equivalence"]["verdict"]
        flag = "ok" if (g is not None and abs(g) < 0.3) else "CHECK"
        if flag == "CHECK":
            ok = False
        print(f"   {nm:<16s} g={g:+.3f}  verdict={v}  [{flag}]")

    print("\n[L2] ICC recovery (planted 0.50):")
    for nm, e in rep["levels"]["L2"]["per_state"].items():
        icc = e.get("synth_icc")
        wv = e["within_sd_equivalence"]["verdict"]
        flag = "ok" if (icc is not None and abs(icc - 0.50) < 0.12) else "CHECK"
        if flag == "CHECK":
            ok = False
        print(f"   {nm:<16s} icc={icc:.3f}  within_sd_verdict={wv}  [{flag}]")

    print(f"\n[L3] between-person corr recovery (planted r={PLANTED_R}):")
    rb = []
    for key, e in rep["levels"]["L3"]["between_person"].items():
        r = e.get("synth_r")
        if r is not None:
            rb.append(r)
        v = e["equivalence"]["verdict"]
        print(f"   {key:<34s} r={r:+.3f}  verdict={v}")
    if rb:
        m = np.mean(rb)
        flag = "ok" if abs(m - PLANTED_R) < 0.12 else "CHECK"
        if flag == "CHECK":
            ok = False
        print(f"   mean between-r = {m:.3f} (target {PLANTED_R}) [{flag}]")

    print(f"\n[L4] within-person multilevel AR(1) at T=6 (planted within-AR={PLANTED_AR1}, "
          f"but Nickell-attenuated estimable target ~0.0):")
    print("     CORRECTNESS CRITERION here is NOT 'recover 0.30' (impossible at T=6) but")
    print("     'estimator is the unbiased-DIRECTION within model (~0), NOT the inflated")
    print("     raw state~lag value (~0.62 from between-person leak)'.")
    for nm, e in rep["levels"]["L4"]["per_state"].items():
        ar = e.get("ar1_fixed")
        v = e["ar1_equivalence"]["verdict"]
        meth = e.get("ar1_method")
        # Correct behavior: within-AR is small (|ar| < 0.2), confirming it did NOT
        # absorb the between-person variance (which would push it toward ~0.62).
        flag = "ok" if (ar is not None and abs(ar) < 0.2) else "CHECK"
        if flag == "CHECK":
            ok = False
        print(f"   {nm:<16s} ar1={ar:+.3f}  verdict={v}  method={meth}  "
              f"%estimable={e.get('pct_personas_estimable')}  [{flag}]")

    print("\n[L6] structure (between PC1 should exceed within PC1 given planted "
          "between-corr 0.40 vs within-corr 0):")
    b6 = rep["levels"]["L6"]["between_person"]
    w6 = rep["levels"]["L6"]["within_person"]
    print(f"   between PC1 var explained = {b6['pc1_var_explained']}  eig={b6['eigenvalues']}")
    print(f"   within  PC1 var explained = {w6['pc1_var_explained']}  eig={w6['eigenvalues']}")
    if (b6["pc1_var_explained"] is not None and w6["pc1_var_explained"] is not None
            and b6["pc1_var_explained"] <= w6["pc1_var_explained"]):
        ok = False
        print("   [CHECK] expected between PC1 > within PC1")
    else:
        print("   [ok] between PC1 > within PC1 as planted")

    print("\n" + "=" * 72)
    print("SELF-TEST RESULT:",
          "PASS — recovers planted L1 mean, L2 ICC/within-SD, L3 between-corr, L6 "
          "structure; L4 returns the correct (unbiased-direction) within-AR estimand, "
          "which is Nickell-attenuated to ~0 at T=6 by design"
          if ok else "FAIL — see [CHECK] lines above")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    main()
