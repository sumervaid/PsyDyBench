"""
Model-swappable LLM client for the generation harness .

A thin wrapper over the OpenAI Python SDK so the SAME pipeline code can hit:
  - open INSTRUCT models via OpenRouter   (base_url=https://openrouter.ai/api/v1)
  - OpenAI reference models via the OpenAI default endpoint (base_url=None)

The original the earlier pipeline stages hard-code `OpenAI()` / `AsyncOpenAI()` inside each stage
class. Here the client is constructed ONCE from a ModelSpec and passed in, which
is what makes the harness model-swappable for the leaderboard.

Also provides MockLLMClient: a deterministic stub that exercises the full code
path (prompt build -> "API" call -> parse) with NO network and NO key, so the
pipeline can be proven end-to-end before an OpenRouter key exists.
"""

import os
import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional

from gen_config import ModelSpec, OUTCOMES


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Real client. Lazily imports the OpenAI SDK so --mock needs no install."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = None  # constructed on first use

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI  # lazy import
        api_key = os.environ.get(self.spec.api_key_env())
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set ${self.spec.api_key_env()} for provider "
                f"'{self.spec.provider}' (model {self.spec.model_id}).")
        kwargs = {"api_key": api_key}
        base = self.spec.base_url()
        if base:
            kwargs["base_url"] = base
        self._client = OpenAI(**kwargs)

    def chat(self, system: str, user: str, max_tokens: int,
             temperature: float) -> LLMResponse:
        """One chat completion. `system` is the cache-friendly static prefix."""
        self._ensure_client()
        kwargs = dict(
            model=self.spec.model_id,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # GPT-5 series (and some reasoning models) only support the default
        # temperature (1); sending another value is a hard 400. Omit it for those.
        if not self.spec.model_id.lower().startswith("gpt-5"):
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        # Some OpenRouter-routed models return a 200 with choices=None/[] or a message
        # with content=None (error body, refusal, reasoning-only). Degrade to "" so the
        # caller's parse-retry + midpoint fallback handles it, instead of a hard
        # 'NoneType not subscriptable' that aborts the unit (and trips the breaker).
        choices = getattr(resp, "choices", None) or []
        text = ""
        if choices:
            msg = getattr(choices[0], "message", None)
            text = (getattr(msg, "content", None) or "") if msg else ""
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class MockLLMClient:
    """
    Deterministic offline stub. No network, no key.

    Produces plausibly-shaped output for each stage so the parsers in the
    stages exercise their real code paths:
      - identity  -> a short portrait blob
      - events    -> MORNING_CONTEXT / HOURLY_EXPERIENCE / ACCUMULATED_NARRATIVE
      - rating    -> a STATES block with 4 in-range numbers, deterministically
                     derived from the persona baseline + day-type so mock output
                     is reproducible and roughly honors the seeded baselines.

    The stage is detected from the system/user content rather than a flag, so the
    same client object can serve all three steps (matching the real client API).
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @staticmethod
    def _seeded_unit(*parts: str) -> float:
        """Deterministic float in [0,1) from string parts."""
        h = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def chat(self, system: str, user: str, max_tokens: int,
             temperature: float) -> LLMResponse:
        low = (system + "\n" + user).lower()

        # --- RATING step (the only step that varies per benchmarked model) ---
        if "states:" in low and "output exactly" in low:
            text = self._mock_rating(system, user)
        # --- EVENTS step ---
        elif "hourly_experience" in low or "today's raw indicators" in low \
                or "transform these" in low:
            text = self._mock_events(user)
        # --- IDENTITY step ---
        else:
            text = self._mock_identity(user)

        return LLMResponse(text=text,
                           input_tokens=len(user) // 4,
                           output_tokens=len(text) // 4)

    def _mock_identity(self, user: str) -> str:
        name = self._extract_name(user)
        return (
            f"{name} is a composite synthetic professional whose measured profile "
            "is honored here. Their baselines and day-to-day variation are treated "
            "as character: some days lift them, some weigh on them, and the spread "
            "between is part of who they are. They notice workload and social tone "
            "first, interpret ambiguity through their disposition, and let strong "
            "events linger. Their worst recent day sat near their observed minimum; "
            "their best brushed their observed maximum. This portrait is intended to "
            "be inhabited so the day can be lived authentically.")

    def _mock_events(self, user: str) -> str:
        name = self._extract_name(user) or "the person"
        day_type = "typical"
        for dt in ("low", "high", "typical"):
            if f"day type: {dt}" in user.lower() or f"({dt} day)" in user.lower():
                day_type = dt
                break
        hours = ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "1:00 PM",
                 "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM"]
        lines = ["MORNING_CONTEXT:",
                 f"{name} arrived to a {day_type} workday, reading the room before settling in.",
                 "", "HOURLY_EXPERIENCE:", ""]
        for h in hours:
            lines += [
                f"{h}:",
                "- Behavioral: working through the queue at a steady pace",
                "- Physical: energy holding, a little screen fatigue building",
                "- Social: a couple of exchanges with peers and the manager",
                "- Psychological: focused, with the usual undercurrent of the day",
                "- Digital: messages and meetings in the normal range",
                "- Life_work: home matters mostly stayed in the background",
                "",
            ]
        lines += ["ACCUMULATED_NARRATIVE:",
                  f"Over the day {name} settled into a {day_type} rhythm; nothing "
                  "broke the texture of how this kind of day usually lands."]
        return "\n".join(lines)

    def _mock_rating(self, system: str, user: str) -> str:
        """Derive in-range, baseline-anchored, day-type-shifted ratings."""
        combined = system + user
        day_type = "typical"
        for dt in ("low", "high", "typical"):
            if f"day type: {dt}" in combined.lower():
                day_type = dt
                break
        shift = {"low": -0.6, "typical": 0.0, "high": 0.6}[day_type]

        out = ["STATES:"]
        for name, cfg in OUTCOMES.items():
            baseline = self._extract_baseline(user, cfg.display_name)
            if baseline is None:
                baseline = (cfg.scale_min + cfg.scale_max) / 2
            # small deterministic jitter tied to persona+state+model
            jitter = (self._seeded_unit(combined[:200], name, self.spec.name) - 0.5)
            val = baseline + shift + jitter
            val = max(cfg.scale_min, min(cfg.scale_max, round(val, 1)))
            out.append(f"- {cfg.display_name}: {val} - mock-derived from baseline "
                       f"{baseline:.1f} on a {day_type} day")
        return "\n".join(out)

    @staticmethod
    def _extract_name(text: str) -> str:
        m = re.search(r"You are\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"portrait of\s+([A-Z][a-z]+)", text)
        return m.group(1) if m else "Alex"

    @staticmethod
    def _extract_baseline(text: str, display_name: str) -> Optional[float]:
        # matches "Work Engagement (1-7): baseline 4.20" / "Your typical range" hints
        m = re.search(
            rf"{re.escape(display_name)}[^\n]*?baseline[^\d]*([0-9]+\.?[0-9]*)",
            text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None


def make_client(spec: ModelSpec, mock: bool = False):
    """Factory: return a mock or real client for the given ModelSpec."""
    if mock or spec.provider == "mock":
        return MockLLMClient(spec)
    return LLMClient(spec)
