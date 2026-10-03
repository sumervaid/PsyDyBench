#!/usr/bin/env python3
"""
seed_personas.py  : forward literature-parametric persona SEEDER.

All-silico replacement for the real-data stage0 seeding. Draws N synthetic personas
from PUBLISHED norms (data/seeds/norms.json) and emits a synthetic CSV in the SAME
80-column schema as `inverse_corpus_prepared.csv`, so it can feed the existing
stage0 / generation pipeline unchanged.

Design (FORWARD, not inverse -> removes the circularity backdoor):
  1. Big Five drawn from a multivariate normal using published means/SDs + the
     meta-analytic intercorrelation matrix.
  2. Demographics drawn from published US working-adult reference distributions.
  3. Each persona's per-state BASELINE (personal mean) is a linear function of
     standardized Big Five (published trait->state betas) PLUS a residual
     between-person term, calibrated so the realized between-person SD matches the
     ICC norm:  between_sd = total_sd * sqrt(ICC).
  4. within-person SD is set directly to the ICC-implied target:
        within_sd = total_sd * sqrt(1 - ICC).
  5. Daily context INDICATORS are sampled forward as inputs. The 4 *_state columns
     are left EMPTY (NaN) for the downstream LLM generation stage to fill. We do NOT
     sample states here -> no inverse leakage.

NO real ESM individual-level data is read. Only norms.json + the schema column order.

Usage:
  python3 seed_personas.py --n 10 --days 6 --seed 42 \
      --out ../../data/seeds/synthetic_personas_seed.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent          # .../persona-psychometrics-aims
DEFAULT_NORMS = REPO_ROOT / "data" / "seeds" / "norms.json"
DEFAULT_OUT = REPO_ROOT / "data" / "seeds" / "synthetic_personas_seed.csv"

# Exact 80-column output schema (order matters: matches inverse_corpus_prepared.csv).
SCHEMA_COLUMNS = [
    "employee_id",
    # behavioral (8)
    "behavioral_task_variety", "behavioral_meeting_hours", "behavioral_deep_work_hours",
    "behavioral_reactive_work_ratio", "behavioral_task_completion_count",
    "behavioral_task_interruption_count", "behavioral_break_minutes",
    "behavioral_overtime_indication",
    # physical (6)
    "physical_avg_energy_level", "physical_energy_trajectory",
    "physical_physical_discomfort_count", "physical_location_changes",
    "physical_movement_level", "physical_commute_stress",
    # social (10)
    "social_total_interaction_count", "social_manager_interaction_minutes",
    "social_manager_interaction_quality", "social_peer_interaction_count",
    "social_support_received_count", "social_support_given_count",
    "social_conflict_instances", "social_isolation_hours",
    "social_meeting_speaking_ratio", "social_social_energy_drain",
    # psychological (10)
    "psychological_stress_level", "psychological_autonomy_felt",
    "psychological_competence_affirmation", "psychological_meaning_connection",
    "psychological_affect_valence", "psychological_affect_arousal",
    "psychological_flow_hours", "psychological_rumination_level",
    "psychological_anticipatory_dread", "psychological_anticipatory_excitement",
    # digital (6)
    "digital_email_volume", "digital_slack_message_volume", "digital_video_meeting_hours",
    "digital_notification_overwhelm", "digital_sync_vs_async_ratio", "digital_screen_fatigue",
    # life_work (15)
    "life_work_sleep_quality", "life_work_sleep_hours", "life_work_morning_mood",
    "life_work_personal_stress_level", "life_work_family_demands", "life_work_health_concerns",
    "life_work_financial_worry", "life_work_major_positive_event",
    "life_work_major_negative_event", "life_work_exercise_today", "life_work_nutrition_quality",
    "life_work_caffeine_intake", "life_work_evening_plans_valence",
    "life_work_weekend_proximity_boost", "life_work_upcoming_life_event",
    # day index + states (states left empty for forward generation)
    "day",
    "work_engagement_state", "psych_safety_state", "work_creativity_state", "team_trust_state",
    "day_of_week", "work_location",
    # Big Five
    "extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness",
    # demographics / context
    "industry", "seniority", "tenure_years", "team_size", "org_culture_creativity",
    "team_interdependence", "age", "sex", "married",
    # trait scores
    "work_engagement_trait", "psychological_safety_trait", "team_trust_trait",
]

STATE_COLS = {
    "work_engagement": "work_engagement_state",
    "psych_safety": "psych_safety_state",
    "work_creativity": "work_creativity_state",
    "team_trust": "team_trust_state",
}
BIG5_ORDER = ["extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness"]


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
def load_norms(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def nearest_pd(corr: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite correlation matrix."""
    sym = (corr + corr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-6, None)
    pd_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # rescale to unit diagonal
    d = np.sqrt(np.diag(pd_mat))
    pd_mat = pd_mat / np.outer(d, d)
    return pd_mat


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def truncnorm_sample(rng, mean, sd, lo, hi, size):
    """Rejection-free truncated normal via clipping resample (good enough for seeding)."""
    out = rng.normal(mean, sd, size)
    # resample tails a few times to reduce clipping pileup, then clip
    for _ in range(5):
        mask = (out < lo) | (out > hi)
        if not mask.any():
            break
        out[mask] = rng.normal(mean, sd, mask.sum())
    return np.clip(out, lo, hi)


def cast_value(x, vtype):
    if vtype == "int":
        return int(round(x))
    if vtype == "binary":
        return int(round(np.clip(x, 0, 1)))
    if vtype == "float1":
        return round(float(x), 1)
    if vtype == "float2":
        return round(float(x), 2)
    return float(x)


# ---------------------------------------------------------------------------
# Persona generation
# ---------------------------------------------------------------------------
def sample_big_five(rng, norms, n):
    """Draw n personas' Big Five from MVN (published means/SDs + intercorr)."""
    bf = norms["big_five"]
    means = np.array([bf["means"][t] for t in BIG5_ORDER])
    sds = np.array([bf["sds"][t] for t in BIG5_ORDER])
    corr = np.array(bf["intercorrelations"]["matrix"], dtype=float)
    corr = nearest_pd(corr)
    cov = corr * np.outer(sds, sds)
    raw = rng.multivariate_normal(means, cov, size=n)
    # clip to trait scale 1-7
    raw = np.clip(raw, 1.0, 7.0)
    return raw, means, sds  # (n,5)


def derive_state_baselines(rng, norms, big5_raw, means, sds):
    """
    For each state, baseline_i = grand_mean + between_sd * standardized_signal_i,
    where standardized_signal combines a trait-driven part (published betas on
    z-scored traits) and a residual between-person part, normalized to unit SD so the
    realized between-person SD equals the ICC target.

    Returns dict[state] -> (baselines[n], within_sd_scalar, between_sd_scalar, icc).
    """
    n = big5_raw.shape[0]
    # z-score traits using POPULATION means/sds (published), not sample stats.
    z = (big5_raw - means) / sds  # (n,5)

    out = {}
    for state, cfg in norms["states"].items():
        gmean = cfg["grand_mean"]
        total_sd = cfg["total_sd"]
        icc = cfg["icc"]
        between_sd = total_sd * np.sqrt(icc)
        within_sd = total_sd * np.sqrt(1.0 - icc)

        betas = np.array([cfg["betas_std"][t] for t in BIG5_ORDER])
        trait_signal = z @ betas  # (n,) in standardized-trait units

        # Variance explained by traits (R^2-ish) given standardized betas & trait corr.
        # Use the realized SD of trait_signal as the trait-driven SD in standardized units.
        trait_sd = trait_signal.std()
        # Residual between-person variance so total standardized between SD = 1.
        resid_var = max(0.0, 1.0 - trait_sd ** 2) if trait_sd < 1.0 else 0.0
        resid = rng.normal(0.0, np.sqrt(resid_var) if resid_var > 0 else 1e-6, n)

        signal = trait_signal + resid
        # Standardize the combined signal to unit SD, then scale to between_sd.
        sig_sd = signal.std() if signal.std() > 0 else 1.0
        signal_std = (signal - signal.mean()) / sig_sd
        baselines = gmean + between_sd * signal_std

        # Clip baselines to construct scale.
        lo, hi = cfg["scale"]
        baselines = np.clip(baselines, lo, hi)

        out[state] = {
            "baselines": baselines,
            "within_sd": within_sd,
            "between_sd": between_sd,
            "icc": icc,
            "trait_signal": trait_signal,
        }
    return out


def sample_demographics(rng, norms, n):
    """Draw person-level demographics from published reference distributions."""
    d = norms["demographics"]
    res = {}

    def draw(spec, key):
        dist = spec["dist"]
        if dist == "categorical":
            vals = spec["values"]
            probs = np.array(spec["probs"], dtype=float)
            probs = probs / probs.sum()
            return rng.choice(vals, size=n, p=probs)
        if dist == "truncnorm":
            x = truncnorm_sample(rng, spec["mean"], spec["sd"], spec["min"], spec["max"], n)
            return np.array([cast_value(v, spec["type"]) for v in x])
        if dist == "lognormal":
            x = rng.lognormal(np.log(spec["median"]), spec["sigma"], n)
            x = np.clip(x, spec["min"], spec["max"])
            return np.array([cast_value(v, spec["type"]) for v in x])
        raise ValueError(f"unknown dist {dist} for {key}")

    for key, spec in d.items():
        res[key] = draw(spec, key)
    return res


def sample_indicators_for_day(rng, norms):
    """Sample one day's worth of the 55 forward-input indicators."""
    inds = norms["indicators"]
    row = {}
    for name, spec in inds.items():
        if name.startswith("_"):
            continue
        lo, hi, vtype = spec
        mean = (lo + hi) / 2.0
        sd = (hi - lo) / 4.0 if hi > lo else 0.0
        if vtype == "binary":
            # Bernoulli at p ~ 0.25 (events/flags are relatively rare by default).
            row[name] = int(rng.random() < 0.25)
        else:
            x = truncnorm_sample(rng, mean, sd, lo, hi, 1)[0]
            row[name] = cast_value(x, vtype)
    return row


def build_dataframe(rng, norms, n, days):
    big5_raw, means, sds = sample_big_five(rng, norms, n)
    state_info = derive_state_baselines(rng, norms, big5_raw, means, sds)
    demo = sample_demographics(rng, norms, n)

    # Person-level trait scores. WE/PS/TT traits exist in schema; set trait = baseline
    # for the matching constructs (stable disposition proxy). Mirrors stage0 behavior.
    rows = []
    for i in range(n):
        emp_id = f"SYN-{i + 1:04d}"
        person = {
            "extraversion": round(float(big5_raw[i, 0]), 2),
            "agreeableness": round(float(big5_raw[i, 1]), 2),
            "conscientiousness": round(float(big5_raw[i, 2]), 2),
            "emotional_stability": round(float(big5_raw[i, 3]), 2),
            "openness": round(float(big5_raw[i, 4]), 2),
            "industry": demo["industry"][i],
            "seniority": demo["seniority"][i],
            "tenure_years": float(demo["tenure_years"][i]),
            "team_size": int(demo["team_size"][i]),
            "org_culture_creativity": float(demo["org_culture_creativity"][i]),
            "team_interdependence": float(demo["team_interdependence"][i]),
            "age": int(demo["age"][i]),
            "sex": int(demo["sex"][i]),
            "married": demo["married"][i],
            # trait scores from baselines (1-7 metric for WE; PS/TT on their construct scales,
            # matching how the real schema stores them as person-level dispositions)
            "work_engagement_trait": round(float(state_info["work_engagement"]["baselines"][i]), 4),
            "psychological_safety_trait": round(float(state_info["psych_safety"]["baselines"][i]), 4),
            "team_trust_trait": round(float(state_info["team_trust"]["baselines"][i]), 4),
        }
        dow_vals = norms["demographics"]["day_of_week"]["values"]
        dow_probs = np.array(norms["demographics"]["day_of_week"]["probs"], float)
        loc_vals = norms["demographics"]["work_location"]["values"]
        loc_probs = np.array(norms["demographics"]["work_location"]["probs"], float)

        for day in range(1, days + 1):
            row = {col: np.nan for col in SCHEMA_COLUMNS}
            row["employee_id"] = emp_id
            row["day"] = day
            row.update(person)
            row.update(sample_indicators_for_day(rng, norms))
            row["day_of_week"] = rng.choice(dow_vals, p=dow_probs / dow_probs.sum())
            row["work_location"] = rng.choice(loc_vals, p=loc_probs / loc_probs.sum())
            # FORWARD: states left NaN for the LLM generation stage to fill.
            row["work_engagement_state"] = np.nan
            row["psych_safety_state"] = np.nan
            row["work_creativity_state"] = np.nan
            row["team_trust_state"] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)[SCHEMA_COLUMNS]
    return df, state_info, big5_raw


def write_sidecar(out_path, norms, n, days, seed, state_info, big5_raw):
    """Write a small JSON sidecar with the per-persona baselines/within-SDs (provenance)."""
    sidecar = {
        "n_personas": n,
        "days_per_persona": days,
        "seed": seed,
        "norms_file": "data/seeds/norms.json",
        "state_targets": {},
        "personas": [],
    }
    for state, info in state_info.items():
        sidecar["state_targets"][state] = {
            "within_sd_target": round(float(info["within_sd"]), 4),
            "between_sd_target": round(float(info["between_sd"]), 4),
            "icc": info["icc"],
            "realized_between_sd": round(float(np.std(info["baselines"])), 4),
        }
    for i in range(n):
        sidecar["personas"].append({
            "employee_id": f"SYN-{i + 1:04d}",
            "big_five": {t: round(float(big5_raw[i, j]), 2) for j, t in enumerate(BIG5_ORDER)},
            "baselines": {s: round(float(state_info[s]["baselines"][i]), 4) for s in state_info},
            "within_sds": {s: round(float(state_info[s]["within_sd"]), 4) for s in state_info},
        })
    side_path = Path(out_path).with_suffix(".targets.json")
    with open(side_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return side_path


def main():
    ap = argparse.ArgumentParser(description="Forward literature-parametric persona seeder (all-silico).")
    ap.add_argument("--n", type=int, default=10, help="Number of synthetic personas.")
    ap.add_argument("--days", type=int, default=6, help="Days per persona.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--norms", type=Path, default=DEFAULT_NORMS, help="Path to norms.json.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    norms = load_norms(args.norms)

    df, state_info, big5_raw = build_dataframe(rng, norms, args.n, args.days)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    side_path = write_sidecar(args.out, norms, args.n, args.days, args.seed, state_info, big5_raw)

    # ---- report ----
    print(f"Wrote {args.out}")
    print(f"  shape: {df.shape}  (expected cols=80)")
    print(f"  personas: {df.employee_id.nunique()}  days/persona: {args.days}")
    print(f"  sidecar: {side_path}")
    print()
    print("State baseline spread (should VARY across personas, by trait):")
    for s, info in state_info.items():
        bl = info["baselines"]
        print(f"  {s:16s}  baseline mean={bl.mean():.3f}  sd(between)={bl.std():.3f} "
              f"(target {info['between_sd']:.3f})  within_sd={info['within_sd']:.3f}  "
              f"min={bl.min():.2f} max={bl.max():.2f}")
    print()
    print("State columns are NaN (forward; filled by generation):",
          df[["work_engagement_state", "psych_safety_state",
              "work_creativity_state", "team_trust_state"]].isna().all().to_dict())


if __name__ == "__main__":
    main()
