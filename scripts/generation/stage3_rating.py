"""
Stage 3 : First-person state rating, model-swappable.

Adapted from an earlier internal implementation (stage3_simulation.py). Changes vs original:
  - This is the ONLY step re-run per benchmarked model (shared-event design):
    identity + day narratives are generated once on a fixed model, then each
    rater model rates the SAME narratives.
  - FIXED UNIFORM rating prompt across all models (NOT per-model DSPy-optimized) so
    the leaderboard is not confounded by prompt tuning  rule 3.
  - Cache-friendly structure: the prompt is assembled as
       [static persona prefix]  +  [static day narrative]  +  [tiny rating instruction]
    The static prefix+narrative is byte-identical across the 4 states for a given
    day (here we rate all 4 in one call), and the persona prefix is byte-identical
    across all 6 days - maximizing prompt-cache reuse on providers that support it.
  - PA/NA removed (project does not predict them). Robust retry + clamp kept.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from gen_config import OUTCOMES, SCALE_INTERPRETATIONS
from stage1_identity import Identity
from stage2_events import NarrativizedDay

SYSTEM_RATING = "You are the person described. Respond in first person, honestly."


@dataclass
class RatedDay:
    employee_id: str
    day_index: int
    day_of_week: str
    day_type: str
    states: Dict[str, float]
    state_reasoning: Dict[str, str] = field(default_factory=dict)
    narrative: str = ""
    metadata: Dict = field(default_factory=dict)


def _scale_guidance(identity: Identity) -> str:
    lines = []
    for name, cfg in OUTCOMES.items():
        rng = identity.observed_ranges.get(name, [cfg.scale_min, cfg.scale_max])
        bl = identity.baselines.get(name, (cfg.scale_min + cfg.scale_max) / 2)
        lines.append(f"\n{cfg.display_name} ({cfg.scale_min:.0f}-{cfg.scale_max:.0f}) "
                     f"- baseline {bl:.2f}, your typical range [{rng[0]:.1f}, {rng[1]:.1f}]:")
        lines.append(cfg.description)
        for val, desc in sorted(SCALE_INTERPRETATIONS.get(name, {}).items()):
            lines.append(f"  {val}: {desc}")
    return "\n".join(lines)


def build_static_prefix(identity: Identity, day: NarrativizedDay) -> str:
    """
    The large, cache-reusable block: persona portrait + today's narrative + the
    (persona-specific but model-invariant) scale guidance. Identical bytes across
    every rater model, so it can be a prompt-cache prefix.
    """
    return (
        f"{identity.persona_prefix}\n"
        f"=== TODAY (Day {day.day_index}, {day.day_of_week}) — DAY TYPE: {day.day_type} ===\n\n"
        f"{day.narrative}\n\n"
        f"=== SCALE GUIDANCE ===\n{_scale_guidance(identity)}\n"
    )


# FIXED UNIFORM rating instruction - identical for EVERY model. Do not tune
# per-model (would confound the capability-vs-fidelity leaderboard).
RATING_INSTRUCTION = """=== RATE YOUR END-OF-DAY STATE ===

You have just lived this day. It is 5:30pm. Report your ACTUAL internal state right
now - not how you should feel, how you do feel. Honor the texture of the day.

OUTPUT EXACTLY this block and nothing else (one number each, within the stated scale):

STATES:
- Work Engagement: [1-7] - [one sentence why]
- Psychological Safety: [1-4] - [one sentence why]
- Work Creativity: [1-6] - [one sentence why]
- Team Trust: [1-5] - [one sentence why]"""


# parse patterns: bare (4.2) or bracketed ([4.2]) numbers
_STATE_PATTERNS: List[Tuple[str, str]] = [
    (r'work\s*engagement[:\s]*\[?(\d+\.?\d*)\]?', 'work_engagement'),
    (r'psychological\s*safety[:\s]*\[?(\d+\.?\d*)\]?', 'psych_safety'),
    (r'psych\s*safety[:\s]*\[?(\d+\.?\d*)\]?', 'psych_safety'),
    (r'work\s*creativity[:\s]*\[?(\d+\.?\d*)\]?', 'work_creativity'),
    (r'team\s*trust[:\s]*\[?(\d+\.?\d*)\]?', 'team_trust'),
]


def parse_states(response: str) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    states: Dict[str, float] = {}
    reasoning: Dict[str, str] = {}
    for line in response.split('\n'):
        low = line.strip().lower()
        for pattern, name in _STATE_PATTERNS:
            if name in states:
                continue
            m = re.search(pattern, low)
            if not m:
                continue
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            cfg = OUTCOMES[name]
            val = max(cfg.scale_min, min(cfg.scale_max, val))
            states[name] = val
            rest = line.strip()[m.end():]
            if '-' in rest:
                reasoning[name] = rest.split('-', 1)[1].strip()
            break
    missing = [s for s in OUTCOMES if s not in states]
    return states, reasoning, missing


def _retry_extract(client, spec, raw_response: str,
                   missing: List[str]) -> Dict[str, float]:
    prompt = ("Extract ONLY the psychological state ratings from the text below. "
              "Return EXACTLY:\n"
              "Work Engagement: [number 1-7]\n"
              "Psychological Safety: [number 1-4]\n"
              "Work Creativity: [number 1-6]\n"
              "Team Trust: [number 1-5]\n\n"
              f"TEXT:\n{raw_response}")
    resp = client.chat("Extract numbers only.", prompt, max_tokens=120, temperature=0.0)
    states, _, _ = parse_states(resp.text or "")
    return {k: v for k, v in states.items() if k in missing}


def rate_day(identity: Identity, day: NarrativizedDay, client, spec) -> RatedDay:
    """Run the rating step for one persona-day on one (benchmarked) model."""
    static_prefix = build_static_prefix(identity, day)
    # cache-friendly: static prefix goes in the system message (reused verbatim),
    # the small fixed instruction is the user message.
    resp = client.chat(static_prefix, RATING_INSTRUCTION,
                       max_tokens=spec.max_tokens_simulation,
                       temperature=spec.temperature_simulation)
    text = resp.text or ""
    states, reasoning, missing = parse_states(text)
    in_tok, out_tok = resp.input_tokens, resp.output_tokens

    if missing:
        retry = _retry_extract(client, spec, text, missing)
        states.update(retry)
        reasoning.update({k: "extracted via retry" for k in retry})
        still = [s for s in missing if s not in states]
        for s in still:  # midpoint fallback
            cfg = OUTCOMES[s]
            states[s] = (cfg.scale_min + cfg.scale_max) / 2
            reasoning[s] = "FALLBACK: unparseable after retry"

    return RatedDay(
        employee_id=identity.employee_id,
        day_index=day.day_index,
        day_of_week=day.day_of_week,
        day_type=day.day_type,
        states=states,
        state_reasoning=reasoning,
        narrative=day.narrative,
        metadata={
            'rater_model': spec.model_id,
            'rater_name': spec.name,
            'tokens': [in_tok, out_tok],
            'n_fallback': len([r for r in reasoning.values() if r.startswith("FALLBACK")]),
            'static_prefix_chars': len(static_prefix),
        },
    )
