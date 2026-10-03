"""Low-ICC daily work states + literature anchors (sourced 2026-06-22, fully cited).
Used by the ICC-headroom experiment: do synthetic within-person effects grow as 1-ICC grows?
Non-affect (PA/NA excluded). M/SD are medium/low confidence (see agent provenance); within_r
anchors are the meta-analytic targets. ICCs verified in the 0.30-0.55 band."""

NEW_STATES = {
    "daily_exhaustion_state": {"min": 1, "max": 7, "display": "Daily Emotional Exhaustion",
        "mean": 2.8, "sd": 1.3, "icc": 0.34,
        "interp": {1: "fully energized, not at all drained", 4: "moderately worn out",
                   7: "completely emotionally drained / used up by work today"}},
    "daily_fatigue_state": {"min": 1, "max": 5, "display": "Daily Fatigue",
        "mean": 2.5, "sd": 0.9, "icc": 0.40,
        "interp": {1: "no fatigue, alert", 3: "somewhat tired", 5: "extremely fatigued, no energy left"}},
    "daily_flow_state": {"min": 1, "max": 7, "display": "Daily Flow / Absorption",
        "mean": 4.5, "sd": 1.1, "icc": 0.48,
        "interp": {1: "never absorbed, no flow today", 4: "occasionally in flow",
                   7: "fully immersed, time flew, carried away by my work"}},
}

NEW_STATE_BENCHMARKS = [
    {'indicator': 'psychological_stress_level', 'state': 'daily_exhaustion_state', 'meta_r': 0.40, 'citation': 'Crawford, LePine & Rich (2010) JAP; Luo et al. (2023) PSPR', 'confidence': 'high'},
    {'indicator': 'behavioral_meeting_hours', 'state': 'daily_exhaustion_state', 'meta_r': 0.25, 'citation': 'Reis et al. (2024) Work & Stress', 'confidence': 'medium'},
    {'indicator': 'life_work_sleep_quality', 'state': 'daily_exhaustion_state', 'meta_r': -0.30, 'citation': 'Litwiller et al. (2017) JAP', 'confidence': 'high'},
    {'indicator': 'social_conflict_instances', 'state': 'daily_exhaustion_state', 'meta_r': 0.28, 'citation': 'Incivility DSEM diary (PubMed 37067475)', 'confidence': 'high'},
    {'indicator': 'behavioral_overtime_indication', 'state': 'daily_exhaustion_state', 'meta_r': 0.22, 'citation': 'Reis et al. (2024)', 'confidence': 'medium'},
    {'indicator': 'social_social_energy_drain', 'state': 'daily_exhaustion_state', 'meta_r': 0.30, 'citation': 'Bakker & Demerouti (2017) JD-R', 'confidence': 'medium'},
    {'indicator': 'psychological_rumination_level', 'state': 'daily_exhaustion_state', 'meta_r': 0.27, 'citation': 'Querstret & Cropley (2012) JOHP', 'confidence': 'high'},
    {'indicator': 'behavioral_break_minutes', 'state': 'daily_exhaustion_state', 'meta_r': -0.18, 'citation': 'Albulescu et al. (2022) PLOS ONE', 'confidence': 'medium'},
    {'indicator': 'life_work_sleep_quality', 'state': 'daily_fatigue_state', 'meta_r': -0.35, 'citation': 'Sleep-quality diaries (PMC10704841; PMC6858829)', 'confidence': 'high'},
    {'indicator': 'life_work_sleep_hours', 'state': 'daily_fatigue_state', 'meta_r': -0.25, 'citation': 'Barber/Munz (PMC6858829)', 'confidence': 'high'},
    {'indicator': 'behavioral_meeting_hours', 'state': 'daily_fatigue_state', 'meta_r': 0.28, 'citation': 'Reis et al. (2024) Work & Stress', 'confidence': 'high'},
    {'indicator': 'physical_avg_energy_level', 'state': 'daily_fatigue_state', 'meta_r': -0.45, 'citation': 'Shirom-Melamed vigor-fatigue bipolarity', 'confidence': 'medium'},
    {'indicator': 'behavioral_overtime_indication', 'state': 'daily_fatigue_state', 'meta_r': 0.25, 'citation': 'Reis et al. (2024)', 'confidence': 'medium'},
    {'indicator': 'life_work_exercise_today', 'state': 'daily_fatigue_state', 'meta_r': -0.18, 'citation': 'Conn et al. (2009) AJPM', 'confidence': 'medium'},
    {'indicator': 'psychological_stress_level', 'state': 'daily_fatigue_state', 'meta_r': 0.32, 'citation': 'Kraiss (2024); Crawford et al. (2010)', 'confidence': 'high'},
    {'indicator': 'behavioral_break_minutes', 'state': 'daily_fatigue_state', 'meta_r': -0.20, 'citation': 'Albulescu et al. (2022)', 'confidence': 'medium'},
    {'indicator': 'psychological_autonomy_felt', 'state': 'daily_flow_state', 'meta_r': 0.35, 'citation': 'Tadic, Bakker & Oerlemans (2015); Bakker (2005)', 'confidence': 'high'},
    {'indicator': 'psychological_competence_affirmation', 'state': 'daily_flow_state', 'meta_r': 0.30, 'citation': 'Bakker (2008) WOLF; Olafsen & Frolund (2018)', 'confidence': 'medium'},
    {'indicator': 'behavioral_deep_work_hours', 'state': 'daily_flow_state', 'meta_r': 0.30, 'citation': 'Csikszentmihalyi; software-flow logs (arXiv:2304.04711)', 'confidence': 'medium'},
    {'indicator': 'behavioral_task_interruption_count', 'state': 'daily_flow_state', 'meta_r': -0.28, 'citation': 'Mark, Gudith & Klocke (2008) CHI', 'confidence': 'medium'},
    {'indicator': 'psychological_flow_hours', 'state': 'daily_flow_state', 'meta_r': 0.55, 'citation': 'manifest convergent indicator (flag: not a within-person causal effect)', 'confidence': 'high'},
    {'indicator': 'social_manager_interaction_quality', 'state': 'daily_flow_state', 'meta_r': 0.22, 'citation': 'Bakker (2005)', 'confidence': 'medium'},
    {'indicator': 'digital_notification_overwhelm', 'state': 'daily_flow_state', 'meta_r': -0.25, 'citation': 'Mark et al. (2008) CHI', 'confidence': 'medium'},
    {'indicator': 'behavioral_task_variety', 'state': 'daily_flow_state', 'meta_r': 0.18, 'citation': 'Bakker (2008) WOLF', 'confidence': 'low'},
]
