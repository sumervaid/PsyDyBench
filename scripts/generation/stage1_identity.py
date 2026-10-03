"""
Stage 1 : Identity portrait generation, model-swappable.

Adapted from an earlier internal implementation (stage1_identity.py). Changes vs original:
  - Operates on a flat `persona` dict (a row of the the pipeline personas CSV) rather
    than the the earlier pipeline EmployeeTargets dataclass, so it needs no an earlier internal pipeline import.
  - LLM client is PASSED IN (LLMClient | MockLLMClient), not constructed via OpenAI().
  - The static (cache-friendly) persona prefix is built once and reused.
"""

from dataclasses import dataclass
from typing import Dict, Any

from gen_config import (OUTCOMES, BIG5_TRAITS, VARIANCE_SIGNATURES,
                        SCALE_INTERPRETATIONS)

SYSTEM_IDENTITY = "You are a clinical organizational psychologist."


@dataclass
class Identity:
    employee_id: str
    first_name: str
    last_name: str
    portrait: str
    reactivity_patterns: str
    observed_ranges: Dict[str, list]
    baselines: Dict[str, float]
    within_sds: Dict[str, float]
    persona_prefix: str  # static, cache-friendly persona block (reused in stage 2/3)
    metadata: Dict[str, Any]


# ---- persona-dict accessors (the pipeline CSV schema) -------------------------
def _f(persona: Dict, key: str, default=None):
    v = persona.get(key, default)
    try:
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return default
        return float(v)
    except (TypeError, ValueError):
        return v


def get_baselines(persona: Dict) -> Dict[str, float]:
    out = {}
    for name, cfg in OUTCOMES.items():
        v = _f(persona, f"baseline_{name}")
        if v is None:
            v = (cfg.scale_min + cfg.scale_max) / 2
        out[name] = v
    return out


def get_within_sds(persona: Dict) -> Dict[str, float]:
    out = {}
    for name in OUTCOMES:
        v = _f(persona, f"within_sd_{name}")
        out[name] = v if v is not None else 0.6
    return out


def get_observed_ranges(persona: Dict) -> Dict[str, list]:
    bl = get_baselines(persona)
    ws = get_within_sds(persona)
    out = {}
    for name, cfg in OUTCOMES.items():
        lo = max(cfg.scale_min, bl[name] - 2 * ws[name])
        hi = min(cfg.scale_max, bl[name] + 2 * ws[name])
        out[name] = [round(lo, 2), round(hi, 2)]
    return out


def _format_big5(persona: Dict) -> str:
    lines = []
    for tname, cfg in BIG5_TRAITS.items():
        val = _f(persona, cfg.column)
        if val is None:
            continue
        if val < 3.0:
            interp = cfg.low_label
        elif val > 5.0:
            interp = cfg.high_label
        else:
            interp = "Moderate"
        lines.append(f"- {cfg.name}: {val:.1f}/7 ({interp})")
    return "\n".join(lines)


def _format_demographics(persona: Dict) -> str:
    lines = []
    age = _f(persona, 'age')
    if age is not None:
        lines.append(f"- Age: {int(age)}")
    sex = _f(persona, 'sex')
    if sex is not None:
        try:
            lines.append(f"- Gender: {'Male' if int(sex) == 1 else 'Female'}")
        except (ValueError, TypeError):
            pass
    for key, label in [('industry', 'Industry'), ('seniority', 'Role Level'),
                       ('married', 'Marital Status')]:
        if persona.get(key):
            lines.append(f"- {label}: {persona[key]}")
    tenure = _f(persona, 'tenure_years')
    if tenure is not None:
        lines.append(f"- Tenure: {tenure:.1f} years")
    return "\n".join(lines)


def _format_patterns(persona: Dict) -> str:
    bl = get_baselines(persona)
    ws = get_within_sds(persona)
    rng = get_observed_ranges(persona)
    lines = []
    for name, cfg in OUTCOMES.items():
        scale_range = cfg.scale_max - cfg.scale_min
        pct = (bl[name] - cfg.scale_min) / scale_range
        bl_interp = "below average" if pct < 0.33 else ("moderate" if pct < 0.67 else "above average")
        lines.append(f"\n{cfg.display_name} (scale: {cfg.scale_min:.0f}-{cfg.scale_max:.0f}):")
        lines.append(f"  - Personal baseline: {bl[name]:.2f} ({bl_interp})")
        lines.append(f"  - Daily variation: SD = {ws[name]:.2f}")
        lines.append(f"  - Observed range: [{rng[name][0]:.1f}, {rng[name][1]:.1f}]")
    return "\n".join(lines)


def _variance_signature(persona: Dict) -> str:
    we_sd = get_within_sds(persona)['work_engagement']
    if we_sd < 0.5:
        return VARIANCE_SIGNATURES['stable']['description']
    if we_sd > 0.9:
        return VARIANCE_SIGNATURES['reactive']['description']
    return VARIANCE_SIGNATURES['moderate']['description']


def _reactivity_patterns(persona: Dict) -> str:
    patterns = []
    es = _f(persona, 'emotional_stability', 4.0)
    if es < 3.5:
        patterns += ["Criticism lands hard and lingers",
                     "Stress accumulates rather than dissipates"]
    elif es > 5.0:
        patterns += ["Takes criticism in stride - processes and moves on",
                     "Good at compartmentalizing stress"]
    else:
        patterns.append("Moderately affected by criticism - notices but doesn't dwell")
    ex = _f(persona, 'extraversion', 4.0)
    if ex < 3.5:
        patterns.append("Social interaction costs energy; needs solitude to recharge")
    elif ex > 5.0:
        patterns.append("Social interaction is energizing; isolation feels draining")
    c = _f(persona, 'conscientiousness', 4.0)
    if c > 5.0:
        patterns.append("Self-worth tied to performance and delivery")
    elif c < 3.5:
        patterns.append("Comfortable with imperfection; doesn't take failures personally")
    a = _f(persona, 'agreeableness', 4.0)
    if a > 5.0:
        patterns.append("Conflict is viscerally uncomfortable; others' emotions are contagious")
    elif a < 3.5:
        patterns.append("Comfortable with professional disagreement")
    return "\n".join(f"- {p}" for p in patterns)


def build_persona_prefix(persona: Dict, portrait: str) -> str:
    """
    Static persona+portrait block reused (verbatim) across all days/states for a
    persona. Keeping it byte-identical is what lets prompt caching kick in on the
    rating step (the only step that re-runs per benchmarked model).
    """
    first = persona.get('first_name', 'Alex')
    last = persona.get('last_name', 'Doe')
    return (
        f"=== WHO YOU ARE ===\n\n"
        f"You are {first} {last}.\n\n"
        f"{portrait}\n\n"
        f"=== YOUR REACTIVITY PATTERNS ===\n{_reactivity_patterns(persona)}\n"
    )


def build_identity_prompt(persona: Dict) -> str:
    first = persona.get('first_name', 'Alex')
    last = persona.get('last_name', 'Doe')
    return f"""You are a clinical organizational psychologist creating a deep psychological portrait.

=== MEASURED DATA (facts - do not contradict) ===

Name: {first} {last}

Demographics:
{_format_demographics(persona)}

Personality (Big Five):
{_format_big5(persona)}

=== OBSERVED PSYCHOLOGICAL PATTERNS ===
{_format_patterns(persona)}

Overall Variance Signature: {_variance_signature(persona)}

=== TASK ===

Create a psychological portrait of {first} (400-600 words) that:
1. EXPLAINS their baselines (why these engagement/safety/creativity/trust levels feel inevitable).
2. EXPLAINS their variance (why their good days are good and bad days bad).
3. DESCRIBES their inner life (worries, aspirations, what threatens and restores them).
4. INCLUDES concrete extreme-day examples at their observed minimum and maximum.
5. CAPTURES their reactivity (how events land and linger, or don't).

Write in third person. Be specific, not generic. DO NOT include any preamble - start with the portrait."""


def generate_identity(persona: Dict, client, spec) -> Identity:
    """Run the identity step once (on the fixed event-generation model)."""
    prompt = build_identity_prompt(persona)
    resp = client.chat(SYSTEM_IDENTITY, prompt,
                       max_tokens=spec.max_tokens_identity,
                       temperature=spec.temperature_identity)
    portrait = (resp.text or "").strip()
    return Identity(
        employee_id=str(persona.get('employee_id', 'UNKNOWN')),
        first_name=persona.get('first_name', 'Alex'),
        last_name=persona.get('last_name', 'Doe'),
        portrait=portrait,
        reactivity_patterns=_reactivity_patterns(persona),
        observed_ranges=get_observed_ranges(persona),
        baselines=get_baselines(persona),
        within_sds=get_within_sds(persona),
        persona_prefix=build_persona_prefix(persona, portrait),
        metadata={
            'identity_model': spec.model_id,
            'word_count': len(portrait.split()),
            'tokens': [resp.input_tokens, resp.output_tokens],
        },
    )
