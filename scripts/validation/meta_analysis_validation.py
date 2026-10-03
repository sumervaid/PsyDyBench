"""
Meta-analytic anchor library and correlation helpers (external-validity framework).

META_BENCHMARKS maps each (indicator, state) pair to a published meta-analytic
correlation with its citation, construct-match note, and confidence tag. It is the
target library for the meta-analytic convergence anchoring leaderboard
(scripts/analysis/anchoring_leaderboard.py) and the within-person magnitude
analyses (scripts/magnitude/eval_within_r.py).

Verification status of each anchor is tracked in data/anchors/ (per-pair audit
JSON) and data/seeds/fulltext_extractions.json; see docs/ANCHOR_SOURCES.md.

The helper functions compute pooled, within-person, and between-person
correlations between one indicator column and one state column of a long
(persona x day) data frame, plus a persona-cluster bootstrap for the pooled r.
"""
import numpy as np
import pandas as pd
from scipy import stats

# Indicator column prefixes (the six indicator dimensions)
DIM_PREFIXES = ['behavioral', 'physical', 'social', 'psychological', 'digital', 'life_work']


# ══════════════════════════════════════════════════════════════════════════════
# META-ANALYTIC LOOKUP TABLE
#
# Each entry maps one (indicator, state) pair to a published meta-analytic r.
# confidence: high  = exact construct match, rho from meta-analytic table
#             medium = close conceptual match or value from abstract/secondary source
#             low   = inferred from related construct
#
# ⚠ Values from web search — verify against originals before publication.
# ══════════════════════════════════════════════════════════════════════════════

META_BENCHMARKS = [
    # ══════════════════════════════════════════════════════════════════════════
    # WORK ENGAGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    {
        'indicator': 'psychological_stress_level',
        'state': 'work_engagement_state',
        'meta_r': -0.19,
        'citation': 'Crawford, LePine & Rich (2010) JAP',
        'construct_match': 'Hindrance demands → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_autonomy_felt',
        'state': 'work_engagement_state',
        'meta_r': 0.39,
        'citation': 'Christian, Garza & Slaughter (2011) Pers Psych',
        'construct_match': 'Autonomy → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'behavioral_task_variety',
        'state': 'work_engagement_state',
        'meta_r': 0.53,
        'citation': 'Christian, Garza & Slaughter (2011) Pers Psych',
        'construct_match': 'Task variety → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_meaning_connection',
        'state': 'work_engagement_state',
        'meta_r': 0.51,
        'citation': 'Christian, Garza & Slaughter (2011) Pers Psych',
        'construct_match': 'Task significance → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'social_support_received_count',
        'state': 'work_engagement_state',
        'meta_r': 0.32,
        'citation': 'Christian, Garza & Slaughter (2011) Pers Psych',
        'construct_match': 'Social support → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_competence_affirmation',
        'state': 'work_engagement_state',
        'meta_r': 0.48,
        'citation': 'Mazzetti et al. (2023) Psych Reports',
        'construct_match': 'Personal resources (self-efficacy) → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_flow_hours',
        'state': 'work_engagement_state',
        'meta_r': 0.50,
        'citation': 'Conceptual: absorption subscale of UWES',
        'construct_match': 'Flow/absorption ≈ Engagement (absorption)',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_deep_work_hours',
        'state': 'work_engagement_state',
        'meta_r': 0.28,
        'citation': 'Christian, Garza & Slaughter (2011) Pers Psych',
        'construct_match': 'Problem solving → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'behavioral_task_interruption_count',
        'state': 'work_engagement_state',
        'meta_r': -0.19,
        'citation': 'Crawford, LePine & Rich (2010) JAP',
        'construct_match': 'Hindrance demands → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_affect_valence',
        'state': 'work_engagement_state',
        'meta_r': 0.62,
        'citation': 'Young et al. (2018) JOB',
        'construct_match': 'Positive affectivity → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_rumination_level',
        'state': 'work_engagement_state',
        'meta_r': -0.25,
        'citation': 'Querstret & Cropley (2012) JOHP (primary study)',
        'construct_match': 'Affective rumination → Fatigue (proxy for disengagement)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_conflict_instances',
        'state': 'work_engagement_state',
        'meta_r': -0.19,
        'citation': 'Crawford, LePine & Rich (2010) JAP',
        'construct_match': 'Role conflict (hindrance demand) → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'social_manager_interaction_quality',
        'state': 'work_engagement_state',
        'meta_r': 0.27,
        'citation': 'Christian, Garza & Slaughter (2011) Pers Psych',
        'construct_match': 'Transformational leadership → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_sleep_quality',
        'state': 'work_engagement_state',
        'meta_r': 0.25,
        'citation': 'Litwiller et al. (2017) JAP (estimated)',
        'construct_match': 'Sleep quality → Work engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'physical_avg_energy_level',
        'state': 'work_engagement_state',
        'meta_r': 0.50,
        'citation': 'Conceptual: vigor subscale of UWES',
        'construct_match': 'Vigor/energy ≈ Engagement (vigor)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_morning_mood',
        'state': 'work_engagement_state',
        'meta_r': 0.62,
        'citation': 'Young et al. (2018) JOB',
        'construct_match': 'Positive affectivity → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_meeting_hours',
        'state': 'work_engagement_state',
        'meta_r': -0.18,
        'citation': 'Rogelberg et al. (2006) JAP',
        'construct_match': 'Meeting load → Engagement (hindrance)',
        'confidence': 'medium',
    },
    {
        'indicator': 'behavioral_overtime_indication',
        'state': 'work_engagement_state',
        'meta_r': -0.15,
        'citation': 'Wong, Chan & Ngan (2019) IJERPH meta-analysis',
        'construct_match': 'Overtime → Engagement (depletion)',
        'confidence': 'medium',
    },
    {
        'indicator': 'behavioral_break_minutes',
        'state': 'work_engagement_state',
        'meta_r': 0.17,
        'citation': 'Albulescu et al. (2022) PLOS ONE meta d=0.36→r',
        'construct_match': 'Breaks → Engagement (recovery)',
        'confidence': 'high',
    },
    {
        'indicator': 'behavioral_reactive_work_ratio',
        'state': 'work_engagement_state',
        'meta_r': -0.20,
        'citation': 'LePine et al. (2005) AMJ hindrance stressor meta',
        'construct_match': 'Reactive/hindrance work → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_family_demands',
        'state': 'work_engagement_state',
        'meta_r': -0.25,
        'citation': 'Amstad et al. (2011) JOHP meta k=427',
        'construct_match': 'Family-work conflict → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'life_work_personal_stress_level',
        'state': 'work_engagement_state',
        'meta_r': -0.22,
        'citation': 'Nohe et al. (2015) JAP panel meta',
        'construct_match': 'Personal stress → Engagement (spillover)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_exercise_today',
        'state': 'work_engagement_state',
        'meta_r': 0.20,
        'citation': 'Conn et al. (2009) AJPM meta',
        'construct_match': 'Exercise → Engagement (vitality)',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_sleep_hours',
        'state': 'work_engagement_state',
        'meta_r': 0.15,
        'citation': 'Henderson & Horan (2021) JOB meta',
        'construct_match': 'Sleep duration → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'digital_notification_overwhelm',
        'state': 'work_engagement_state',
        'meta_r': -0.22,
        'citation': 'Sarabadani et al. (2023) EJIS technostress meta',
        'construct_match': 'Notification overload → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'digital_screen_fatigue',
        'state': 'work_engagement_state',
        'meta_r': -0.20,
        'citation': 'Sarabadani et al. (2023) techno-strain',
        'construct_match': 'Screen fatigue → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'digital_email_volume',
        'state': 'work_engagement_state',
        'meta_r': -0.18,
        'citation': 'Schmitt et al. (2021) info overload meta',
        'construct_match': 'Email overload → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'physical_commute_stress',
        'state': 'work_engagement_state',
        'meta_r': -0.15,
        'citation': 'Murphy et al. (2023) Org Psych Rev meta',
        'construct_match': 'Commute stress → Engagement',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_affect_arousal',
        'state': 'work_engagement_state',
        'meta_r': 0.35,
        'citation': 'Young et al. (2018) JOB / circumplex model',
        'construct_match': 'Activated affect → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_anticipatory_dread',
        'state': 'work_engagement_state',
        'meta_r': -0.25,
        'citation': 'Crawford, LePine & Rich (2010) JAP (hindrance appraisal)',
        'construct_match': 'Anticipatory dread → Disengagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_anticipatory_excitement',
        'state': 'work_engagement_state',
        'meta_r': 0.35,
        'citation': 'Crawford, LePine & Rich (2010) JAP (challenge appraisal)',
        'construct_match': 'Challenge anticipation → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'social_isolation_hours',
        'state': 'work_engagement_state',
        'meta_r': -0.20,
        'citation': 'Halbesleben (2010) JOB meta: social resources',
        'construct_match': 'Social isolation → Engagement (inverse resource)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_peer_interaction_count',
        'state': 'work_engagement_state',
        'meta_r': 0.25,
        'citation': 'Halbesleben (2010) JOB meta: coworker support',
        'construct_match': 'Peer interaction → Engagement',
        'confidence': 'medium',
    },
    {
        'indicator': 'social_social_energy_drain',
        'state': 'work_engagement_state',
        'meta_r': -0.22,
        'citation': 'Bakker & Demerouti (2017) JOB: emotional demands',
        'construct_match': 'Social energy drain → Engagement (depletion)',
        'confidence': 'low',
    },
    {
        'indicator': 'physical_energy_trajectory',
        'state': 'work_engagement_state',
        'meta_r': 0.30,
        'citation': 'Sonnentag (2003) JAP: recovery → engagement',
        'construct_match': 'Rising energy → Engagement (vigor)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_support_given_count',
        'state': 'work_engagement_state',
        'meta_r': 0.15,
        'citation': 'Halbesleben (2010) JOB: prosocial behavior ↔ engagement',
        'construct_match': 'Giving support → Engagement (reciprocal resource)',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_task_completion_count',
        'state': 'work_engagement_state',
        'meta_r': 0.22,
        'citation': 'Bakker & Bal (2010) JOHP: daily task resources',
        'construct_match': 'Task accomplishment → Engagement',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_financial_worry',
        'state': 'work_engagement_state',
        'meta_r': -0.15,
        'citation': 'Schieman & Young (2011) J Health Soc Behavior',
        'construct_match': 'Financial stress → Engagement (distraction)',
        'confidence': 'low',
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PSYCHOLOGICAL SAFETY
    # ══════════════════════════════════════════════════════════════════════════
    {
        'indicator': 'psychological_autonomy_felt',
        'state': 'psych_safety_state',
        'meta_r': 0.47,
        'citation': 'Frazier et al. (2017) Pers Psych',
        'construct_match': 'Autonomy → Psychological safety',
        'confidence': 'high',
    },
    {
        'indicator': 'social_support_received_count',
        'state': 'psych_safety_state',
        'meta_r': 0.49,
        'citation': 'Frazier et al. (2017) Pers Psych',
        'construct_match': 'Supportive work context → Psychological safety',
        'confidence': 'high',
    },
    {
        'indicator': 'social_manager_interaction_quality',
        'state': 'psych_safety_state',
        'meta_r': 0.42,
        'citation': 'Frazier et al. (2017) Pers Psych',
        'construct_match': 'Transformational leadership → Psychological safety',
        'confidence': 'high',
    },
    {
        'indicator': 'social_conflict_instances',
        'state': 'psych_safety_state',
        'meta_r': -0.30,
        'citation': 'Frazier et al. (2017) Pers Psych (inferred from role clarity .63)',
        'construct_match': 'Interpersonal conflict → Psychological safety (inverse)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_stress_level',
        'state': 'psych_safety_state',
        'meta_r': -0.25,
        'citation': 'Inferred from Frazier et al. (2017) supportive context (.49)',
        'construct_match': 'Stress → Psychological safety (inverse)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_isolation_hours',
        'state': 'psych_safety_state',
        'meta_r': -0.30,
        'citation': 'Inferred: social isolation ≈ inverse of supportive context',
        'construct_match': 'Social isolation → Psychological safety (inverse)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_affect_valence',
        'state': 'psych_safety_state',
        'meta_r': 0.35,
        'citation': 'Frazier et al. (2017): satisfaction ↔ psych safety ρ=.53',
        'construct_match': 'Positive affect → Psychological safety (proxy via satisfaction)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_competence_affirmation',
        'state': 'psych_safety_state',
        'meta_r': 0.40,
        'citation': 'Frazier et al. (2017): supportive context (.49), autonomy (.47)',
        'construct_match': 'Competence → Psychological safety (personal resource)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_sleep_quality',
        'state': 'psych_safety_state',
        'meta_r': 0.15,
        'citation': 'Litwiller et al. (2017) JAP (estimated, indirect path)',
        'construct_match': 'Sleep quality → General well-being → Psych safety',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_meaning_connection',
        'state': 'psych_safety_state',
        'meta_r': 0.35,
        'citation': 'Frazier et al. (2017): learning behavior (.62), commitment (.48)',
        'construct_match': 'Meaning/purpose → Psychological safety (via commitment path)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_peer_interaction_count',
        'state': 'psych_safety_state',
        'meta_r': 0.30,
        'citation': 'Frazier et al. (2017): supportive work context (.49)',
        'construct_match': 'Peer interaction → Psychological safety (social resource)',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_rumination_level',
        'state': 'psych_safety_state',
        'meta_r': -0.20,
        'citation': 'Querstret & Cropley (2012) + Frazier et al. (2017) inferred',
        'construct_match': 'Rumination → Psychological safety (negative affect path)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_social_energy_drain',
        'state': 'psych_safety_state',
        'meta_r': -0.31,
        'citation': 'Profit et al. (2024) Health Affairs Scholar',
        'construct_match': 'Social energy drain → Psychological safety',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_family_demands',
        'state': 'psych_safety_state',
        'meta_r': -0.16,
        'citation': 'Bilal et al. (2020) Front Psych',
        'construct_match': 'Family demands → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'digital_notification_overwhelm',
        'state': 'psych_safety_state',
        'meta_r': -0.15,
        'citation': 'Sarabadani et al. (2023) chained path',
        'construct_match': 'Notification overload → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_meeting_hours',
        'state': 'psych_safety_state',
        'meta_r': 0.12,
        'citation': 'Allen & Rogelberg (2013) Group & Org Mgmt',
        'construct_match': 'Meeting interaction → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_exercise_today',
        'state': 'psych_safety_state',
        'meta_r': 0.10,
        'citation': 'Conn et al. (2009) indirect via well-being',
        'construct_match': 'Exercise → Psychological safety (well-being path)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_affect_arousal',
        'state': 'psych_safety_state',
        'meta_r': 0.15,
        'citation': 'Frazier et al. (2017) indirect: engagement → safety',
        'construct_match': 'Activated positive affect → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_personal_stress_level',
        'state': 'psych_safety_state',
        'meta_r': -0.20,
        'citation': 'Nohe et al. (2015) JAP: personal stress spillover',
        'construct_match': 'Personal stress → Psychological safety (spillover)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_anticipatory_dread',
        'state': 'psych_safety_state',
        'meta_r': -0.30,
        'citation': 'Edmondson (1999) ASQ: fear/anxiety inverse of psych safety',
        'construct_match': 'Dread/anxiety → Psychological safety (inverse)',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_anticipatory_excitement',
        'state': 'psych_safety_state',
        'meta_r': 0.20,
        'citation': 'Edmondson & Lei (2014) Annu Rev Org Psych: positive anticipation',
        'construct_match': 'Positive anticipation → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_morning_mood',
        'state': 'psych_safety_state',
        'meta_r': 0.25,
        'citation': 'Beal et al. (2005) AMR: affective events → workplace behavior',
        'construct_match': 'Morning mood → Psychological safety (affect priming)',
        'confidence': 'low',
    },
    {
        'indicator': 'physical_avg_energy_level',
        'state': 'psych_safety_state',
        'meta_r': 0.15,
        'citation': 'Resource theory: energy → interpersonal risk capacity',
        'construct_match': 'Energy → Psychological safety (resource)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_sleep_hours',
        'state': 'psych_safety_state',
        'meta_r': 0.12,
        'citation': 'Barnes et al. (2015) AMJ: sleep → interpersonal functioning',
        'construct_match': 'Sleep hours → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'social_support_given_count',
        'state': 'psych_safety_state',
        'meta_r': 0.25,
        'citation': 'Frazier et al. (2017): reciprocal support → safety climate',
        'construct_match': 'Giving support → Psychological safety (reciprocity)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_total_interaction_count',
        'state': 'psych_safety_state',
        'meta_r': 0.20,
        'citation': 'Frazier et al. (2017): supportive work context (.49)',
        'construct_match': 'Social interaction volume → Psychological safety',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_task_variety',
        'state': 'psych_safety_state',
        'meta_r': 0.15,
        'citation': 'Spreitzer (1995) AMJ: job enrichment → empowerment → safety',
        'construct_match': 'Task variety → Psychological safety (empowerment path)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_financial_worry',
        'state': 'psych_safety_state',
        'meta_r': -0.12,
        'citation': 'Probst (2005) JOHP: financial insecurity → reduced safety climate',
        'construct_match': 'Financial worry → Psychological safety (threat)',
        'confidence': 'low',
    },

    # ══════════════════════════════════════════════════════════════════════════
    # WORK CREATIVITY
    # ══════════════════════════════════════════════════════════════════════════
    {
        'indicator': 'psychological_autonomy_felt',
        'state': 'work_creativity_state',
        'meta_r': 0.32,
        'citation': 'Liu et al. (2016) JOB meta-analysis',
        'construct_match': 'Autonomy → Creativity',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_affect_valence',
        'state': 'work_creativity_state',
        'meta_r': 0.15,
        'citation': 'Baas, De Dreu & Nijstad (2008) Psych Bull meta',
        'construct_match': 'Positive affect → Creativity',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_meaning_connection',
        'state': 'work_creativity_state',
        'meta_r': 0.36,
        'citation': 'Hammond et al. (2011) JAP meta',
        'construct_match': 'Intrinsic motivation/meaning → Creativity',
        'confidence': 'high',
    },
    {
        'indicator': 'social_support_received_count',
        'state': 'work_creativity_state',
        'meta_r': 0.21,
        'citation': 'Hammond et al. (2011) JAP meta',
        'construct_match': 'Social support → Creativity',
        'confidence': 'high',
    },
    {
        'indicator': 'social_manager_interaction_quality',
        'state': 'work_creativity_state',
        'meta_r': 0.21,
        'citation': 'Hammond et al. (2011) JAP meta',
        'construct_match': 'Transformational leadership → Creativity',
        'confidence': 'medium',
    },
    {
        'indicator': 'behavioral_task_variety',
        'state': 'work_creativity_state',
        'meta_r': 0.17,
        'citation': 'Hammond et al. (2011) JAP meta (job complexity)',
        'construct_match': 'Task complexity/variety → Creativity',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_stress_level',
        'state': 'work_creativity_state',
        'meta_r': -0.10,
        'citation': 'Byron, Khazanchi & Nazarian (2010) JAP meta',
        'construct_match': 'Hindrance stress → Creativity',
        'confidence': 'high',
    },
    {
        'indicator': 'social_conflict_instances',
        'state': 'work_creativity_state',
        'meta_r': -0.12,
        'citation': 'De Dreu & Weingart (2003) JAP meta',
        'construct_match': 'Relationship conflict → Creativity/performance',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_competence_affirmation',
        'state': 'work_creativity_state',
        'meta_r': 0.30,
        'citation': 'Tierney & Farmer (2002) AMJ + Liu et al. (2016)',
        'construct_match': 'Creative self-efficacy → Creativity',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_flow_hours',
        'state': 'work_creativity_state',
        'meta_r': 0.40,
        'citation': 'Csikszentmihalyi (1996) conceptual; Cseh et al. (2015)',
        'construct_match': 'Flow state → Creative output',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_deep_work_hours',
        'state': 'work_creativity_state',
        'meta_r': 0.25,
        'citation': 'Hammond et al. (2011) JAP meta (cognitive engagement)',
        'construct_match': 'Concentrated cognitive work → Creativity',
        'confidence': 'low',
    },
    {
        'indicator': 'physical_avg_energy_level',
        'state': 'work_creativity_state',
        'meta_r': 0.25,
        'citation': 'Binnewies & Woernlein (2011) JOHP; resource theory',
        'construct_match': 'Energy/vitality → Creativity',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_rumination_level',
        'state': 'work_creativity_state',
        'meta_r': -0.15,
        'citation': 'Cropley & Cropley (2016); cognitive depletion path',
        'construct_match': 'Rumination → Creativity (depletion)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_affect_arousal',
        'state': 'work_creativity_state',
        'meta_r': 0.20,
        'citation': 'Baas, De Dreu & Nijstad (2008) Psych Bull meta',
        'construct_match': 'Activating affect → Creativity',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_sleep_quality',
        'state': 'work_creativity_state',
        'meta_r': 0.15,
        'citation': 'Killgore (2010) Progress in Brain Research',
        'construct_match': 'Sleep quality → Divergent thinking',
        'confidence': 'low',
    },
    {
        'indicator': 'behavioral_task_interruption_count',
        'state': 'work_creativity_state',
        'meta_r': -0.15,
        'citation': 'Madjar, Oldham & Pratt (2002) AMJ; hindrance path',
        'construct_match': 'Interruptions → Creativity (disrupted flow)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_isolation_hours',
        'state': 'work_creativity_state',
        'meta_r': -0.15,
        'citation': 'Perry-Smith (2006) AMJ: social networks → creativity',
        'construct_match': 'Social isolation → Creativity (reduced stimulation)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_peer_interaction_count',
        'state': 'work_creativity_state',
        'meta_r': 0.18,
        'citation': 'Perry-Smith & Shalley (2003) AMR: social interaction → ideas',
        'construct_match': 'Peer interaction → Creativity (idea cross-pollination)',
        'confidence': 'medium',
    },
    {
        'indicator': 'social_social_energy_drain',
        'state': 'work_creativity_state',
        'meta_r': -0.18,
        'citation': 'Amabile et al. (2005) ASQ: emotional exhaustion → creativity',
        'construct_match': 'Social drain → Creativity (depletion)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_morning_mood',
        'state': 'work_creativity_state',
        'meta_r': 0.20,
        'citation': 'Amabile et al. (2005) ASQ: daily affect → creativity',
        'construct_match': 'Morning mood → Creativity (affective priming)',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_exercise_today',
        'state': 'work_creativity_state',
        'meta_r': 0.15,
        'citation': 'Oppezzo & Schwartz (2014) JEP:LMC: exercise → divergent thinking',
        'construct_match': 'Exercise → Creativity (cognitive flexibility)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_personal_stress_level',
        'state': 'work_creativity_state',
        'meta_r': -0.15,
        'citation': 'Byron, Khazanchi & Nazarian (2010) JAP: uncontrollable stress',
        'construct_match': 'Personal stress → Creativity (threat rigidity)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_anticipatory_excitement',
        'state': 'work_creativity_state',
        'meta_r': 0.25,
        'citation': 'Baas, De Dreu & Nijstad (2008) Psych Bull: activating positive mood',
        'construct_match': 'Positive anticipation → Creativity',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_anticipatory_dread',
        'state': 'work_creativity_state',
        'meta_r': -0.20,
        'citation': 'Byron, Khazanchi & Nazarian (2010) JAP: anxiety → creativity',
        'construct_match': 'Dread/anxiety → Creativity (threat rigidity)',
        'confidence': 'medium',
    },
    {
        'indicator': 'behavioral_break_minutes',
        'state': 'work_creativity_state',
        'meta_r': 0.15,
        'citation': 'Sio & Ormerod (2009) Psych Bull meta: incubation effect',
        'construct_match': 'Breaks → Creativity (incubation)',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_sleep_hours',
        'state': 'work_creativity_state',
        'meta_r': 0.12,
        'citation': 'Killgore (2010) Progress in Brain Research: sleep → flexibility',
        'construct_match': 'Sleep hours → Creativity (cognitive resource)',
        'confidence': 'low',
    },
    {
        'indicator': 'digital_notification_overwhelm',
        'state': 'work_creativity_state',
        'meta_r': -0.18,
        'citation': 'Mark, Gudith & Klocke (2008) CHI: interruptions → creative flow',
        'construct_match': 'Notification overload → Creativity (flow disruption)',
        'confidence': 'low',
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TEAM TRUST
    # ══════════════════════════════════════════════════════════════════════════
    {
        'indicator': 'social_manager_interaction_quality',
        'state': 'team_trust_state',
        'meta_r': 0.46,
        'citation': 'Dirks & Ferrin (2002) JAP meta',
        'construct_match': 'Transformational leadership → Trust',
        'confidence': 'high',
    },
    {
        'indicator': 'social_support_received_count',
        'state': 'team_trust_state',
        'meta_r': 0.40,
        'citation': 'Dirks & Ferrin (2002) JAP meta',
        'construct_match': 'Perceived org support → Trust',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_autonomy_felt',
        'state': 'team_trust_state',
        'meta_r': 0.30,
        'citation': 'Dirks & Ferrin (2002) JAP meta (participative decision making)',
        'construct_match': 'Autonomy/voice → Trust',
        'confidence': 'medium',
    },
    {
        'indicator': 'social_conflict_instances',
        'state': 'team_trust_state',
        'meta_r': -0.33,
        'citation': 'De Jong & Elfring (2010) AMJ; Dirks & Ferrin (2002)',
        'construct_match': 'Interpersonal conflict → Trust',
        'confidence': 'medium',
    },
    {
        'indicator': 'psychological_competence_affirmation',
        'state': 'team_trust_state',
        'meta_r': 0.35,
        'citation': 'Colquitt, Scott & LePine (2007) JAP meta (ability trust)',
        'construct_match': 'Competence → Trust (ability-based)',
        'confidence': 'high',
    },
    {
        'indicator': 'psychological_affect_valence',
        'state': 'team_trust_state',
        'meta_r': 0.30,
        'citation': 'Colquitt, Scott & LePine (2007) JAP meta',
        'construct_match': 'Positive affect → Trust propensity',
        'confidence': 'low',
    },
    {
        'indicator': 'social_peer_interaction_count',
        'state': 'team_trust_state',
        'meta_r': 0.35,
        'citation': 'Dirks & Ferrin (2002) JAP meta (interaction frequency)',
        'construct_match': 'Peer interaction → Trust (familiarity)',
        'confidence': 'medium',
    },
    {
        'indicator': 'social_isolation_hours',
        'state': 'team_trust_state',
        'meta_r': -0.30,
        'citation': 'Inferred: isolation ≈ inverse of interaction/familiarity',
        'construct_match': 'Social isolation → Trust (inverse)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_stress_level',
        'state': 'team_trust_state',
        'meta_r': -0.20,
        'citation': 'Colquitt, Scott & LePine (2007) JAP meta (trust → stress path)',
        'construct_match': 'Stress → Trust erosion',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_meaning_connection',
        'state': 'team_trust_state',
        'meta_r': 0.25,
        'citation': 'Dirks & Ferrin (2002): commitment ↔ trust ρ=.49',
        'construct_match': 'Meaning → Trust (via commitment)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_social_energy_drain',
        'state': 'team_trust_state',
        'meta_r': -0.28,
        'citation': 'Inferred from Dirks & Ferrin (2002) interpersonal risk path',
        'construct_match': 'Social drain → Trust erosion',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_rumination_level',
        'state': 'team_trust_state',
        'meta_r': -0.18,
        'citation': 'Colquitt et al. (2007); negative cognition path',
        'construct_match': 'Rumination → Trust (negative appraisal)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_sleep_quality',
        'state': 'team_trust_state',
        'meta_r': 0.12,
        'citation': 'Barnes et al. (2015) AMJ; sleep → interpersonal functioning',
        'construct_match': 'Sleep quality → Trust (interpersonal capacity)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_affect_arousal',
        'state': 'team_trust_state',
        'meta_r': 0.15,
        'citation': 'Jones & George (1998) AMR: emotional state → trust formation',
        'construct_match': 'Activated affect → Trust (engagement path)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_support_given_count',
        'state': 'team_trust_state',
        'meta_r': 0.30,
        'citation': 'Dirks & Ferrin (2002) JAP: benevolence behaviors → trust',
        'construct_match': 'Giving support → Trust (benevolence signal)',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_morning_mood',
        'state': 'team_trust_state',
        'meta_r': 0.18,
        'citation': 'Mislin et al. (2015) OBH meta: affect → trust decisions',
        'construct_match': 'Morning mood → Trust (affective priming)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_anticipatory_dread',
        'state': 'team_trust_state',
        'meta_r': -0.22,
        'citation': 'Dunn & Schweitzer (2005) AMJ: anxiety → reduced trust',
        'construct_match': 'Dread/anxiety → Trust (threat appraisal)',
        'confidence': 'medium',
    },
    {
        'indicator': 'social_total_interaction_count',
        'state': 'team_trust_state',
        'meta_r': 0.28,
        'citation': 'Dirks & Ferrin (2002) JAP: familiarity/interaction → trust',
        'construct_match': 'Interaction volume → Trust (familiarity)',
        'confidence': 'medium',
    },
    {
        'indicator': 'life_work_family_demands',
        'state': 'team_trust_state',
        'meta_r': -0.12,
        'citation': 'Spillover theory: family demands → reduced relational investment',
        'construct_match': 'Family demands → Trust (reduced social capacity)',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_personal_stress_level',
        'state': 'team_trust_state',
        'meta_r': -0.18,
        'citation': 'Colquitt, Scott & LePine (2007) JAP: stress → trust propensity',
        'construct_match': 'Personal stress → Trust (reduced propensity)',
        'confidence': 'low',
    },
    {
        'indicator': 'physical_avg_energy_level',
        'state': 'team_trust_state',
        'meta_r': 0.15,
        'citation': 'Resource theory: energy → interpersonal engagement capacity',
        'construct_match': 'Energy → Trust (relational resource)',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_anticipatory_excitement',
        'state': 'team_trust_state',
        'meta_r': 0.18,
        'citation': 'Dunn & Schweitzer (2005) AMJ: positive emotions → trust',
        'construct_match': 'Positive anticipation → Trust',
        'confidence': 'low',
    },
    {
        'indicator': 'life_work_sleep_hours',
        'state': 'team_trust_state',
        'meta_r': 0.10,
        'citation': 'Barnes et al. (2015) AMJ: sleep → interpersonal functioning',
        'construct_match': 'Sleep hours → Trust',
        'confidence': 'low',
    },
    {
        'indicator': 'psychological_flow_hours',
        'state': 'team_trust_state',
        'meta_r': 0.15,
        'citation': 'Salanova et al. (2006) JOHP: flow → collective efficacy → trust',
        'construct_match': 'Flow → Trust (shared positive experience)',
        'confidence': 'low',
    },
    {
        'indicator': 'social_manager_interaction_minutes',
        'state': 'team_trust_state',
        'meta_r': 0.30,
        'citation': 'Dirks & Ferrin (2002) JAP: leader interaction frequency → trust',
        'construct_match': 'Manager time → Trust (relationship investment)',
        'confidence': 'medium',
    },
]


# ── Nice names ───────────────────────────────────────────────────────────────
NICE = {
    'stress_level': 'Stress', 'flow_hours': 'Flow Hours',
    'rumination_level': 'Rumination', 'autonomy_felt': 'Autonomy',
    'affect_valence': 'Mood Valence', 'competence_affirmation': 'Competence',
    'meaning_connection': 'Meaning', 'anticipatory_dread': 'Dread',
    'deep_work_hours': 'Deep Work', 'meeting_hours': 'Meeting Hours',
    'task_interruption_count': 'Interruptions', 'break_minutes': 'Breaks',
    'avg_energy_level': 'Energy', 'energy_trajectory': 'Energy Trend',
    'isolation_hours': 'Isolation', 'conflict_instances': 'Conflict',
    'support_received_count': 'Support Rcvd', 'social_energy_drain': 'Social Drain',
    'peer_interaction_count': 'Peer Interactions', 'manager_interaction_quality': 'Mgr Quality',
    'notification_overwhelm': 'Notif. Overwhelm', 'screen_fatigue': 'Screen Fatigue',
    'sleep_quality': 'Sleep Quality', 'morning_mood': 'Morning Mood',
    'personal_stress_level': 'Personal Stress', 'family_demands': 'Family Demands',
    'sleep_hours': 'Sleep Hours', 'exercise_today': 'Exercise',
    'task_variety': 'Task Variety', 'affect_arousal': 'Arousal',
    'total_interaction_count': 'Total Interactions',
    'reactive_work_ratio': 'Reactive Work', 'anticipatory_excitement': 'Excitement',
    'overtime_indication': 'Overtime', 'commute_stress': 'Commute Stress',
    'email_volume': 'Email Volume', 'task_completion_count': 'Task Completions',
    'slack_message_volume': 'Slack Messages', 'financial_worry': 'Financial Worry',
    'manager_interaction_minutes': 'Mgr Minutes', 'support_given_count': 'Support Given',
    'health_concerns': 'Health Concerns', 'nutrition_quality': 'Nutrition',
    'major_negative_event': 'Neg. Life Event', 'major_positive_event': 'Pos. Life Event',
    'evening_plans_valence': 'Evening Plans', 'caffeine_intake': 'Caffeine',
    'location_changes': 'Location Changes', 'physical_discomfort_count': 'Discomfort',
    'movement_level': 'Movement',
}

STATE_SHORT = {
    'work_engagement_state': 'Engage',
    'psych_safety_state': 'PsySaf',
    'work_creativity_state': 'Creat',
    'team_trust_state': 'Trust',
}


def nice(col):
    """Pretty-print an indicator column name."""
    for pfx in DIM_PREFIXES:
        if col.startswith(pfx + '_'):
            short = col[len(pfx)+1:]
            return NICE.get(short, short.replace('_', ' ').title())
    return col.replace('_', ' ').title()


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_pooled_r(df, indicator, state):
    """Pearson r (pooled, ignoring clustering)."""
    valid = df[[indicator, state]].dropna()
    if len(valid) < 30:
        return np.nan
    return valid[indicator].corr(valid[state])


def compute_within_r(df, indicator, state):
    """Within-person r (person-mean-centered)."""
    sub = df[['employee_id', indicator, state]].dropna()
    if len(sub) < 30:
        return np.nan
    pm_ind = sub.groupby('employee_id')[indicator].transform('mean')
    pm_st  = sub.groupby('employee_id')[state].transform('mean')
    dev_ind = sub[indicator] - pm_ind
    dev_st  = sub[state] - pm_st
    r, _ = stats.pearsonr(dev_ind, dev_st)
    return r


def compute_between_r(df, indicator, state):
    """Between-person r: correlate person-MEANS across persons. THIS is the estimand
    the published meta-analyses report (cross-sectional, across people). compute_pooled_r
    blends levels; compute_within_r is day-to-day. Use this for like-for-like comparison
    to between-person meta_r."""
    sub = df[['employee_id', indicator, state]].dropna()
    if sub['employee_id'].nunique() < 30:
        return np.nan
    pm = sub.groupby('employee_id')[[indicator, state]].mean()
    if pm[indicator].std() == 0 or pm[state].std() == 0:
        return np.nan
    return pm[indicator].corr(pm[state])


def bootstrap_r(df, indicator, state, n_boot=1000, seed=42):
    """Bootstrap 95% CI for pooled Pearson r."""
    rng = np.random.default_rng(seed)
    valid = df[[indicator, state]].dropna()
    n = len(valid)
    if n < 30:
        return np.nan, np.nan
    x = valid[indicator].values
    y = valid[state].values
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
