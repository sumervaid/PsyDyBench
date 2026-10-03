"""
Generation config for the open-model benchmark harness .

Self-contained copy of the locked psychometric facts (4 states, scales, scale
anchors, Big Five labels) so the harness does NOT import anything from the
an earlier internal pipeline tree. The originals live in
and were copied-and-adapted here per the pipeline (DO NOT modify originals).

Locked facts: WE(1-7), PS(1-4), WC(1-6), TT(1-5). Big Five are stable
inputs, not rated. PA/NA are NOT predicted in this project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Provider / endpoint routing
# ---------------------------------------------------------------------------
# Open instruct models are served through OpenRouter's OpenAI-compatible API.
# OpenAI reference models (e.g. gpt-4.1-mini) go to the OpenAI default endpoint
# directly (NOT via OpenRouter) so the reference points are clean.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = None  # None => OpenAI SDK default (https://api.openai.com/v1)


@dataclass
class ModelSpec:
    """Identifies a benchmarked model and how to reach it."""
    name: str                       # short label used in output paths / leaderboard
    model_id: str                   # provider model string passed to the API
    provider: str = "openrouter"    # "openrouter" | "openai" | "mock"
    temperature_identity: float = 0.7
    temperature_simulation: float = 0.8
    max_tokens_identity: int = 4000
    max_tokens_events: int = 4000
    max_tokens_simulation: int = 2000

    def base_url(self) -> Optional[str]:
        if self.provider == "openrouter":
            return OPENROUTER_BASE_URL
        if self.provider == "openai":
            return OPENAI_BASE_URL
        if self.provider == "mock":
            return None
        raise ValueError(f"Unknown provider: {self.provider}")

    def api_key_env(self) -> str:
        if self.provider == "openrouter":
            return "OPENROUTER_API_KEY"
        if self.provider == "openai":
            return "OPENAI_API_KEY"
        return "NONE"


# Example open INSTRUCT roster (the paper = INSTRUCT only; NO base models).
# These ids are OpenRouter model strings; adjust before the real run.
EXAMPLE_ROSTER: List[ModelSpec] = [
    ModelSpec("LLAMA31_8B", "meta-llama/llama-3.1-8b-instruct", "openrouter"),
    ModelSpec("LLAMA31_70B", "meta-llama/llama-3.1-70b-instruct", "openrouter"),
    ModelSpec("QWEN25_7B", "qwen/qwen-2.5-7b-instruct", "openrouter"),
    ModelSpec("QWEN25_72B", "qwen/qwen-2.5-72b-instruct", "openrouter"),
    ModelSpec("MISTRAL_SMALL", "mistralai/mistral-small", "openrouter"),
    ModelSpec("GEMMA2_9B", "google/gemma-2-9b-it", "openrouter"),
    # OpenAI reference point (routed directly to OpenAI, NOT OpenRouter):
    ModelSpec("GPT41_MINI", "gpt-4.1-mini", "openai"),
]


# ---------------------------------------------------------------------------
# The 4 states (locked). Scales: WE 1-7, PS 1-4, WC 1-6, TT 1-5.
# ---------------------------------------------------------------------------
@dataclass
class OutcomeConfig:
    name: str
    display_name: str
    scale_min: float
    scale_max: float
    state_col: str
    description: str = ""


OUTCOMES: Dict[str, OutcomeConfig] = {
    'work_engagement': OutcomeConfig(
        'work_engagement', 'Work Engagement', 1.0, 7.0, 'work_engagement_state',
        "How absorbed, dedicated, and vigorous you feel about your work"),
    'psych_safety': OutcomeConfig(
        'psych_safety', 'Psychological Safety', 1.0, 4.0, 'psych_safety_state',
        "How safe you feel to take risks, speak up, make mistakes without punishment"),
    'work_creativity': OutcomeConfig(
        'work_creativity', 'Work Creativity', 1.0, 6.0, 'work_creativity_state',
        "How creative, generative, and innovative you feel"),
    'team_trust': OutcomeConfig(
        'team_trust', 'Team Trust', 1.0, 5.0, 'team_trust_state',
        "How much you trust your team members"),
}


SCALE_INTERPRETATIONS: Dict[str, Dict[int, str]] = {
    'work_engagement': {
        1: "Completely disengaged - mentally checked out, counting minutes",
        2: "Largely disengaged - going through motions, minimal investment",
        3: "Somewhat disengaged - present but not engaged, doing minimum required",
        4: "Neutral - neither engaged nor disengaged, functional but flat",
        5: "Somewhat engaged - interested, finding some meaning, time moves okay",
        6: "Highly engaged - absorbed, dedicated, time flows naturally",
        7: "Completely engaged - deeply immersed, energized, work feels purposeful",
    },
    'psych_safety': {
        1: "Unsafe - feel I must hide mistakes, can't speak up, fear judgment",
        2: "Mostly unsafe - cautious, guarded, selective about what I share",
        3: "Moderately safe - can speak up sometimes, still some caution",
        4: "Fully safe - can be myself, take risks, mistakes are learning opportunities",
    },
    'work_creativity': {
        1: "No creativity - stuck in routine, no new ideas, just executing",
        2: "Minimal creativity - rare sparks, mostly standard work",
        3: "Some creativity - occasional new ideas, some problem-solving",
        4: "Moderate creativity - generating ideas, making novel connections",
        5: "High creativity - ideas flowing, innovative thinking",
        6: "Peak creativity - breakthrough thinking, deeply generative",
    },
    'team_trust': {
        1: "No trust - wary, protective, doubt competence and intentions",
        2: "Low trust - skeptical, verify everything, limited reliance",
        3: "Neutral trust - professional baseline, neither suspicious nor confident",
        4: "Good trust - comfortable relying on team, believe in good intentions",
        5: "Exceptionally high trust - would rely on them for critical tasks",
    },
}


# Six context dimensions (HCSE structure) used to give each day texture.
INDICATOR_DIMENSIONS: Dict[str, List[str]] = {
    'behavioral': ['task_variety', 'meeting_hours', 'deep_work_hours',
                   'reactive_work_ratio', 'task_completion_count',
                   'task_interruption_count', 'break_minutes', 'overtime_indication'],
    'physical': ['avg_energy_level', 'energy_trajectory', 'physical_discomfort_count',
                 'location_changes', 'movement_level', 'commute_stress'],
    'social': ['total_interaction_count', 'manager_interaction_minutes',
               'manager_interaction_quality', 'peer_interaction_count',
               'support_received_count', 'support_given_count', 'conflict_instances',
               'isolation_hours', 'meeting_speaking_ratio', 'social_energy_drain'],
    'psychological': ['stress_level', 'autonomy_felt', 'competence_affirmation',
                      'meaning_connection', 'affect_valence', 'affect_arousal',
                      'flow_hours', 'rumination_level', 'anticipatory_dread',
                      'anticipatory_excitement'],
    'digital': ['email_volume', 'slack_message_volume', 'video_meeting_hours',
                'notification_overwhelm', 'sync_vs_async_ratio', 'screen_fatigue'],
    'life_work': ['sleep_quality', 'sleep_hours', 'morning_mood', 'personal_stress_level',
                  'family_demands', 'health_concerns', 'financial_worry',
                  'major_positive_event', 'major_negative_event', 'exercise_today',
                  'nutrition_quality', 'caffeine_intake', 'evening_plans_valence',
                  'weekend_proximity_boost', 'upcoming_life_event'],
}

INDICATOR_COLS: List[str] = [
    f"{dim}_{ind}" for dim, inds in INDICATOR_DIMENSIONS.items() for ind in inds
]


@dataclass
class Big5Trait:
    name: str
    column: str
    low_label: str
    high_label: str


BIG5_TRAITS: Dict[str, Big5Trait] = {
    'extraversion': Big5Trait('Extraversion', 'extraversion',
        'Introverted - prefers solitude, finds social interaction draining',
        'Extraverted - energized by people, seeks social stimulation'),
    'agreeableness': Big5Trait('Agreeableness', 'agreeableness',
        'Low agreeableness - competitive, skeptical, challenges others',
        'High agreeableness - cooperative, trusting, harmony-seeking'),
    'conscientiousness': Big5Trait('Conscientiousness', 'conscientiousness',
        'Low conscientiousness - flexible, spontaneous, less detail-oriented',
        'High conscientiousness - disciplined, organized, achievement-focused'),
    'emotional_stability': Big5Trait('Emotional Stability', 'emotional_stability',
        'Neurotic - emotionally reactive, stress-prone, anxiety-sensitive',
        'Emotionally stable - calm under pressure, resilient, even-keeled'),
    'openness': Big5Trait('Openness to Experience', 'openness',
        'Low openness - practical, conventional, prefers routine',
        'High openness - curious, creative, seeks novelty and ideas'),
}


VARIANCE_SIGNATURES: Dict[str, Dict] = {
    'stable': {'description': 'Your experience is relatively consistent day-to-day. '
                              'Events affect you but not dramatically.'},
    'moderate': {'description': 'Your experience has texture but not drama. Good days '
                                'lift you; hard days weigh on you.'},
    'reactive': {'description': 'Your internal experience varies substantially '
                                'day-to-day. You ride the waves.'},
}


@dataclass
class RunConfig:
    """Top-level run knobs (model-independent)."""
    n_days_per_persona: int = 6
    # variance coverage across the n days (low / typical / high day types)
    day_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.15, 'typical': 0.70, 'high': 0.15})
    max_concurrent_requests: int = 5
    seed: int = 17
