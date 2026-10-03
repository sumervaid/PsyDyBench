"""
Stage 2 : Hourly day-narrative generation, model-swappable.

Adapted from an earlier internal implementation (stage2_events.py). Changes vs original:
  - All-silico: there is NO retrieval from a real corpus. Each day's variance
    "type" (low/typical/high) is planned from the persona's own variance profile,
    and the LLM narrativizes a workday of that intensity through the persona's
    psychology. (The original retrieved real observed days as templates; we cannot
    use real data in this repo.)
  - LLM client is PASSED IN.
  - In SHARED-EVENT mode these narratives are generated ONCE (one fixed model) and
    reused as the static prefix for every benchmarked rater.
"""

import random
from dataclasses import dataclass
from typing import Dict, List

from gen_config import RunConfig
from stage1_identity import Identity

SYSTEM_EVENTS = "You transform workplace context into narrative experience."

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


@dataclass
class NarrativizedDay:
    employee_id: str
    day_index: int
    day_of_week: str
    day_type: str        # 'low' | 'typical' | 'high'
    narrative: str       # full hourly narrative text (reused as static prefix)
    metadata: Dict


def plan_variance_coverage(n_days: int, rc: RunConfig, rng: random.Random) -> List[str]:
    """Plan day types so the n days span the persona's variance range."""
    dist = rc.day_type_distribution
    day_types: List[str] = []
    remaining = n_days
    if n_days >= 4:
        day_types += ['low', 'high']
        remaining -= 2
    while remaining > 0:
        r = rng.random()
        if r < dist['low']:
            day_types.append('low')
        elif r < dist['low'] + dist['high']:
            day_types.append('high')
        else:
            day_types.append('typical')
        remaining -= 1
    rng.shuffle(day_types)
    return day_types


def _day_intensity_guidance(day_type: str, identity: Identity) -> str:
    rng = identity.observed_ranges['work_engagement']
    if day_type == 'low':
        return (f"This is one of {identity.first_name}'s HARDER days - toward the "
                f"low end of their range (engagement near {rng[0]:.1f}). Let friction, "
                f"fatigue, or interpersonal strain show, without melodrama.")
    if day_type == 'high':
        return (f"This is one of {identity.first_name}'s BETTER days - toward the "
                f"high end of their range (engagement near {rng[1]:.1f}). Let momentum, "
                f"flow, or connection show.")
    return (f"This is a TYPICAL day for {identity.first_name} - mid-range, the ordinary "
            f"texture of their work life.")


_DIM_LABELS = {"behavioral": "Behavioral", "physical": "Physical", "social": "Social",
               "psychological": "Psychological", "digital": "Digital", "life_work": "Life/Work"}

def _format_indicators(indicators) -> str:
    """Render the day's sampled 55-indicator context, grouped by dimension, so the
    narrative is ANCHORED to these specific values (this is what lets indicator->state
    effects emerge — the day's measured context drives the experience)."""
    if not indicators:
        return ""
    groups = {k: [] for k in _DIM_LABELS}
    for col, val in indicators.items():
        dim = col.split("_")[0] if not col.startswith("life_work") else "life_work"
        if dim not in groups:
            dim = "life_work" if col.startswith("life") else dim
        if dim in groups and val not in (None, "", "nan"):
            name = col.split("_", 1)[1] if dim != "life_work" else col[len("life_work_"):]
            groups[dim].append(f"{name.replace('_', ' ')}={val}")
    lines = []
    for dim, items in groups.items():
        if items:
            lines.append(f"- {_DIM_LABELS[dim]}: " + "; ".join(items))
    return "\n".join(lines)


def build_events_prompt(identity: Identity, day_index: int,
                        day_of_week: str, day_type: str, indicators=None) -> str:
    ctx = _format_indicators(indicators)
    ctx_block = (f"=== TODAY'S MEASURED CONTEXT (anchor the day to THESE values) ===\n{ctx}\n\n"
                 if ctx else "")
    task_line = (
        "The day's measured context is given above. Render how THIS person experiences "
        "THESE specific conditions hour by hour — the narrative must be consistent with "
        "the measured values (e.g., high stress / low autonomy / many meetings should show)."
        if ctx else
        f"Decide what plausibly happens on a {day_type} day for THIS person, then describe "
        "how they experience it.")
    return f"""{identity.persona_prefix}

=== TODAY (Day {day_index}, {day_of_week}) — DAY TYPE: {day_type} ===

{_day_intensity_guidance(day_type, identity)}

{ctx_block}=== TASK ===

Generate the texture of {identity.first_name}'s workday across six dimensions
(behavioral, physical, social, psychological, digital, life_work), filtered through
their psychology. {task_line}

Output format:

MORNING_CONTEXT:
[How {identity.first_name} woke up and arrived at work, through their psychology.]

HOURLY_EXPERIENCE:

9:00 AM:
- Behavioral: [...]
- Physical: [...]
- Social: [...]
- Psychological: [...]
- Digital: [...]
- Life_work: [...]

10:00 AM:
[Continue each hour through 5:00 PM]

ACCUMULATED_NARRATIVE:
[50-100 words on how the day built up emotionally for {identity.first_name}.]

Be specific, not generic. This person's psychology shapes everything."""


def narrativize_day(identity: Identity, day_index: int, day_type: str,
                    client, spec, indicators=None) -> NarrativizedDay:
    day_of_week = DAYS_OF_WEEK[(day_index - 1) % len(DAYS_OF_WEEK)]
    prompt = build_events_prompt(identity, day_index, day_of_week, day_type, indicators)
    resp = client.chat(SYSTEM_EVENTS, prompt,
                       max_tokens=spec.max_tokens_events,
                       temperature=spec.temperature_simulation)
    return NarrativizedDay(
        employee_id=identity.employee_id,
        day_index=day_index,
        day_of_week=day_of_week,
        day_type=day_type,
        narrative=(resp.text or "").strip(),
        metadata={'events_model': spec.model_id,
                  'tokens': [resp.input_tokens, resp.output_tokens]},
    )
