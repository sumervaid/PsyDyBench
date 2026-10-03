#!/usr/bin/env python3
"""
Consolidate FULL-TEXT-VERIFIED meta-analytic correlations from the 8 author-supplied PDFs into a
structured extraction + provenance doc. Values were read by hand from the cited correlation TABLES
(line refs into literature/source_text/*.txt, which are gitignored copies of the publisher PDFs).

Each record carries BOTH the observed sample-size-weighted mean r and the corrected rho where the
paper reports them, plus k, N, the table location, a construct-match quality, and status. The
distinction matters: our simulated correlations are UNCORRECTED, so the observed r is the appropriate
comparison criterion; many original "claimed" anchors turn out to equal the CORRECTED rho.

Outputs: data/seeds/fulltext_extractions.json, results/anchors/FULLTEXT_EXTRACTED_VALUES.md
Reproduce: python3 scripts/scraping/build_fulltext_extractions.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (our_indicator, our_state, paper_construct, r_obs, rho, k, N, table_line, match, status, claimed)
R = lambda *a: dict(zip(
    ["our_indicator","our_state","paper_construct","r_observed","rho_corrected","k","N",
     "source_loc","match_quality","status","claimed"], a))

PAPERS = {
 "Frazier2017": {"cite":"Frazier, Fainshmidt, Klinger, Pezeshkan & Vracheva (2017), Personnel Psychology 70:113-165",
   "doi":"10.1111/peps.12183", "txt":"Frazier2017_psych_safety_meta.txt", "table":"Table 2 (antecedents/outcomes of psychological safety)",
   "pairs":[
    R("psychological_autonomy_felt","psych_safety","Autonomy",.38,.47,8,1661,"L851","close","verified",.47),
    R("psychological_competence_affirmation","psych_safety","Role clarity",.48,.63,6,1017,"L853","loose","verified",.40),
    R("psychological_meaning_connection","psych_safety","Work design characteristics (composite)",.41,.53,26,5768,"L850","loose","verified",.35),
    R("social_support_received_count","psych_safety","Supportive work context",.40,.49,24,5045,"L854","close","verified",.49),
    R("social_peer_interaction_count","psych_safety","Peer support",.50,.62,6,1293,"L855","close","verified",.30),
    R("social_support_given_count","psych_safety","Supportive work context",.40,.49,24,5045,"L854","loose","verified",.25),
    R("psychological_affect_valence","psych_safety","(positive affect not isolated in main table)",None,None,None,None,"-","not_in_paper","not_found",.35),
    R("psychological_affect_arousal","psych_safety","(arousal absent)",None,None,None,None,"-","not_in_paper","not_found",.15),
    R("psychological_stress_level","psych_safety","(role stressors not in antecedent table)",None,None,None,None,"-","not_in_paper","not_found",-.25),
    R("social_conflict_instances","psych_safety","(interpersonal conflict not in main antecedent table)",None,None,None,None,"-","not_in_paper","not_found",-.30),
    R("social_total_interaction_count","psych_safety","(generic interaction absent)",None,None,None,None,"-","not_in_paper","not_found",.20),
    R("physical_avg_energy_level","psych_safety","(energy/vigor absent)",None,None,None,None,"-","not_in_paper","not_found",.15),
    R("life_work_personal_stress_level","psych_safety","(life stress absent)",None,None,None,None,"-","not_in_paper","not_found",-.20),
    R("life_work_sleep_hours","psych_safety","(sleep absent)",None,None,None,None,"-","not_in_paper","not_found",.12),
    R("life_work_sleep_quality","psych_safety","(sleep absent)",None,None,None,None,"-","not_in_paper","not_found",.15),
   ],
   "cross_state":[
    R("psych_safety","work_engagement","Engagement (outcome)",.36,.45,13,3676,"L857","exact","verified",.50),
    R("psych_safety","team_trust","Trust in leadership (antecedent; trust-in-leader != team trust)",.32,.39,6,1280,"L849","loose","verified",.45),
    R("psych_safety","work_creativity","(creativity/innovation not an outcome in this meta)",None,None,None,None,"-","not_in_paper","not_found",.13),
   ]},

 "Crawford2010": {"cite":"Crawford, LePine & Rich (2010), J Applied Psychology 95(5):834-848",
   "doi":"10.1037/a0019364","txt":"Crawford2010_jdr_meta.txt","table":"Tables (engagement correlations with demands/resources)",
   "pairs":[
    R("psychological_autonomy_felt","work_engagement","Autonomy (resource)",.31,.37,32,18344,"L710","close","verified",.39),
    R("psychological_anticipatory_excitement","work_engagement","Challenge stressors",.30,.36,54,27200,"L615","close","verified",.35),
    R("psychological_anticipatory_dread","work_engagement","Hindrance stressors",-.16,-.19,28,18540,"L615","close","verified",-.25),
    R("psychological_stress_level","work_engagement","Role conflict / hindrance stressors",-.20,-.24,12,3689,"L733","close","verified",-.19),
    R("social_support_given_count","work_engagement","Social support (resource)",None,None,None,None,"L640","loose","abstract_only",.15),
    R("social_conflict_instances","work_engagement","Interpersonal conflict (hindrance)",-.20,-.24,12,3689,"L733","loose","verified",-.19),
    R("physical_avg_energy_level","work_engagement","(vigor ~ engagement itself; circular)",None,None,None,None,"-","not_in_paper","not_found",.50),
    R("psychological_flow_hours","work_engagement","(flow absent)",None,None,None,None,"-","not_in_paper","not_found",.50),
    R("behavioral_task_completion_count","work_engagement","(performance not antecedent)",None,None,None,None,"-","not_in_paper","not_found",.22),
    R("behavioral_task_interruption_count","work_engagement","(interruptions absent)",None,None,None,None,"-","not_in_paper","not_found",-.19),
    R("life_work_exercise_today","work_engagement","(exercise absent)",None,None,None,None,"-","not_in_paper","not_found",.20),
   ],"cross_state":[]},

 "DirksFerrin2002": {"cite":"Dirks & Ferrin (2002), J Applied Psychology 87(4):611-628",
   "doi":"10.1037/0021-9010.87.4.611","txt":"DirksFerrin2002_trust_leadership_meta.txt","table":"Table 2 (antecedents of trust in leadership)",
   "pairs":[
    R("social_manager_interaction_quality","team_trust","Transformational leadership (trust in leader)",.72,None,13,5657,"L597","loose","verified",.46),
    R("social_support_received_count","team_trust","Perceived organizational support",.69,None,6,847,"L604","loose","verified",.40),
    R("psychological_autonomy_felt","team_trust","Participative decision making",None,None,None,None,"-","loose","abstract_only",.30),
    R("social_manager_interaction_minutes","team_trust","(interaction quantity absent)",None,None,None,None,"-","not_in_paper","not_found",.30),
    R("social_support_given_count","team_trust","(support given absent)",None,None,None,None,"-","not_in_paper","not_found",.30),
    R("psychological_meaning_connection","team_trust","(meaning absent)",None,None,None,None,"-","not_in_paper","not_found",.25),
    R("social_peer_interaction_count","team_trust","(peer; paper is trust-in-LEADER)",None,None,None,None,"-","not_in_paper","not_found",.35),
    R("social_total_interaction_count","team_trust","(absent)",None,None,None,None,"-","not_in_paper","not_found",.28),
    R("social_social_energy_drain","team_trust","(absent)",None,None,None,None,"-","not_in_paper","not_found",-.28),
    R("physical_avg_energy_level","team_trust","(absent)",None,None,None,None,"-","not_in_paper","not_found",.15),
   ],"cross_state":[]},

 "Colquitt2007": {"cite":"Colquitt, Scott & LePine (2007), J Applied Psychology 92(4):909-927",
   "doi":"10.1037/0021-9010.92.4.909","txt":"Colquitt2007_trust_meta.txt","table":"Table 2 (trustworthiness dimensions -> trust)",
   "pairs":[
    R("psychological_competence_affirmation","team_trust","Ability (trustee competence)",.55,.67,18,3885,"L478","loose","verified",.35),
    R("psychological_affect_valence","team_trust","Trust-affect / affective commitment",None,None,None,None,"L627","loose","abstract_only",.30),
    R("psychological_stress_level","team_trust","(stress absent)",None,None,None,None,"-","not_in_paper","not_found",-.20),
    R("psychological_rumination_level","team_trust","(rumination absent)",None,None,None,None,"-","not_in_paper","not_found",-.18),
    R("psychological_flow_hours","team_trust","(flow absent)",None,None,None,None,"-","not_in_paper","not_found",.15),
    R("life_work_personal_stress_level","team_trust","(life stress absent)",None,None,None,None,"-","not_in_paper","not_found",-.18),
   ],"cross_state":[]},

 "Kleine2019": {"cite":"Kleine, Rudolph & Zacher (2019), J Organizational Behavior 40:973-999",
   "doi":"10.1002/job.2375","txt":"Kleine2019_thriving_meta.txt","table":"abstract + correlate tables (thriving at work)",
   "pairs":[
    R("social_manager_interaction_quality","work_engagement","Supportive leadership -> thriving (thriving~engagement)",None,.44,None,None,"L35","loose","verified",.27),
    R("social_manager_interaction_quality","work_creativity","Thriving -> creativity/innovation",None,None,None,None,"-","loose","abstract_only",.21),
    R("life_work_family_demands","work_engagement","Work-family conflict -> thriving (negative)",None,None,None,None,"-","loose","abstract_only",-.25),
    R("psychological_stress_level","work_creativity","(stress->creativity not focal)",None,None,None,None,"-","not_in_paper","not_found",-.10),
    R("psychological_rumination_level","psych_safety","(rumination absent)",None,None,None,None,"-","not_in_paper","not_found",-.20),
    R("psychological_affect_arousal","team_trust","(absent)",None,None,None,None,"-","not_in_paper","not_found",.15),
    R("life_work_family_demands","team_trust","(absent)",None,None,None,None,"-","not_in_paper","not_found",-.12),
   ],"cross_state":[]},

 "Bennett2017": {"cite":"Bennett, Bakker & Field (2017), J Organizational Behavior 39:262-275",
   "doi":"10.1002/job.2217","txt":"Bennett2017_recovery_meta.txt","table":"Table 2 (demands -> recovery experiences)",
   "pairs":[
    R("behavioral_overtime_indication","work_engagement","Hindrance demands -> psychological detachment (proxy)",-.18,-.21,11,5602,"L405","loose","verified",-.15),
    R("life_work_personal_stress_level","work_engagement","Challenge demands -> detachment (recovery)",-.30,-.37,14,3463,"L400","loose","verified",-.22),
    R("behavioral_break_minutes","work_engagement","(break time -> recovery not focal)",None,None,None,None,"-","not_in_paper","not_found",.17),
    R("physical_commute_stress","work_engagement","(commute absent)",None,None,None,None,"-","not_in_paper","not_found",-.15),
   ],"cross_state":[]},

 "Creativity2015": {"cite":"Tsai (2015) 2nd-order meta / Revista de Psicologia del Trabajo, Personal factors of creativity",
   "doi":"10.1016/j.rpto.2015.06.002","txt":"2015_creativity_2ndorder_meta.txt","table":"abstract + 2nd-order table (personal factors of creativity)",
   "pairs":[
    R("psychological_affect_arousal","work_creativity","Positive affect",.19,None,None,None,"L30","close","verified",.20),
    R("psychological_competence_affirmation","work_creativity","Self-efficacy",.13,None,None,None,"L31","close","verified",.30),
    R("physical_avg_energy_level","work_creativity","(energy not a factor)",None,None,None,None,"-","not_in_paper","not_found",.25),
    R("social_support_received_count","work_creativity","(social support not in this personal-factors meta)",None,None,None,None,"-","not_in_paper","not_found",.21),
   ],"cross_state":[]},

 "Morrison2021": {"cite":"Morrison et al. / Distinguishing Voice and Silence at Work (2021), AMJ 64(1):114-148",
   "doi":"10.5465/amj.2018.1428","txt":"MorrisonVoiceSilence2021.txt","table":"correlation table (voice/silence antecedents)",
   "pairs":[
    R("social_manager_interaction_quality","psych_safety","(psych safety as voice antecedent; loose)",None,None,None,None,"-","loose","abstract_only",.42),
    R("behavioral_meeting_hours","psych_safety","(meeting time absent)",None,None,None,None,"-","not_in_paper","not_found",.12),
   ],"cross_state":[]},
}

def main():
    out = {"_meta":{"purpose":"Full-text-verified meta-analytic correlations from author-supplied PDFs.",
            "method":"hand-read from cited correlation TABLES; line refs into gitignored literature/source_text/*.txt",
            "note":"r_observed = sample-size-weighted mean (uncorrected) = the right criterion for our uncorrected "
                   "simulated correlations; rho_corrected = artifact-corrected. Many original 'claimed' anchors equal "
                   "the CORRECTED rho, which inflates them relative to observed r."},
           "papers":PAPERS}
    (REPO/"data/seeds/fulltext_extractions.json").write_text(json.dumps(out, indent=2))

    # provenance MD
    L=["# Full-text-verified meta-analytic correlations (author-supplied PDFs, 2026-06-20)","",
       "Values hand-read from the cited correlation tables. PDFs live in `literature/source_papers/` "
       "(gitignored, copyright); text dumps in `literature/source_text/` (gitignored). For each pair we "
       "report the OBSERVED sample-size-weighted mean r (the criterion for our uncorrected simulated data) "
       "AND the corrected rho where given.","",
       "**Key provenance finding:** several original anchor values equal the *corrected* rho, not the observed r "
       "(e.g., autonomy->psych_safety claimed 0.47 = Frazier's rho-hat; observed r = .38). Recommend anchoring "
       "internal-validity and convergence targets on OBSERVED r.",""]
    nv=nf=0
    for pid,p in PAPERS.items():
        L.append(f"## {p['cite']}\nDOI {p['doi']} · {p['table']}\n")
        L.append("| our pair | paper construct | obs r | corr rho | k | N | loc | match | status |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in p["pairs"]+p.get("cross_state",[]):
            tgt = r.get("our_state");
            pair=f"{r['our_indicator']} -> {tgt}"
            L.append(f"| {pair} | {r['paper_construct']} | {r['r_observed']} | {r['rho_corrected']} | "
                     f"{r['k']} | {r['N']} | {r['source_loc']} | {r['match_quality']} | {r['status']} |")
            nv += r["status"]=="verified"; nf += r["status"]=="not_found"
        L.append("")
    L.append(f"\n**Totals:** {nv} verified from full-text tables, {nf} not in the paper (no value; honest null).")
    (REPO/"results/anchors/FULLTEXT_EXTRACTED_VALUES.md").write_text("\n".join(L)+"\n")
    print(f"verified={nv} not_found={nf}")
    print("wrote data/seeds/fulltext_extractions.json + results/anchors/FULLTEXT_EXTRACTED_VALUES.md")

if __name__=="__main__":
    main()
