# Anchor sources: the meta-analyses behind the PsyDyBench targets

author-supplied full-text PDFs used to verify the benchmark's meta-analytic targets (2026-06-20).

## Copyright: PDFs and full text stay out of the release
`literature/source_papers/` (publisher PDFs) and `literature/source_text/` (pdftotext dumps) are
**gitignored**: they are copyrighted publisher content (HBS/EBSCO/Wiley access). Only the *extracted
numeric values* + provenance are committed. The files live locally on this machine.

## Papers (8) and what each anchors
| file | citation | DOI | anchors |
|---|---|---|---|
| Frazier2017_psych_safety_meta | Frazier et al. (2017) Pers. Psych. 70:113-165 | 10.1111/peps.12183 | psych-safety antecedents + PS↔engagement / PS↔trust cross-state |
| Crawford2010_jdr_meta | Crawford, LePine & Rich (2010) JAP 95:834-848 | 10.1037/a0019364 | engagement ~ challenge/hindrance/resources |
| DirksFerrin2002_trust_leadership_meta | Dirks & Ferrin (2002) JAP 87:611-628 | 10.1037/0021-9010.87.4.611 | trust-in-leader antecedents |
| Colquitt2007_trust_meta | Colquitt, Scott & LePine (2007) JAP 92:909-927 | 10.1037/0021-9010.92.4.909 | trustworthiness → trust |
| Kleine2019_thriving_meta | Kleine, Rudolph & Zacher (2019) JOB 40:973-999 | 10.1002/job.2375 | thriving correlates |
| Bennett2017_recovery_meta | Bennett, Bakker & Field (2017) JOB 39:262-275 | 10.1002/job.2217 | demands → recovery experiences |
| 2015_creativity_2ndorder_meta | Personal factors of creativity, 2nd-order meta (2015) | 10.1016/j.rpto.2015.06.002 | personal factors → creativity |
| MorrisonVoiceSilence2021 | Distinguishing Voice and Silence at Work (2021) AMJ 64:114-148 | 10.5465/amj.2018.1428 | voice/silence ~ psych safety |

## Extracted values (committed)
- `data/seeds/fulltext_extractions.json`: structured, per pair, observed r + corrected ρ + k + N + table
  location + construct-match quality + status. Built by `scripts/scraping/build_fulltext_extractions.py`.
- `results/anchors/FULLTEXT_EXTRACTED_VALUES.md`: human-readable table and the corrected-vs-observed finding.
- `data/seeds/cross_state_targets.json`: PS↔engagement and PS↔trust upgraded to `full_text_verified`.

## Headline provenance finding
**21 pairs verified** from full-text tables, and **35 honest nulls** (the construct is absent from
the paper, so the pair keeps its prior). Several original "claimed" anchors equal the papers' *corrected* ρ rather than
the *observed* r (e.g., autonomy→psych-safety claimed 0.47 = Frazier's ρ̂, observed r = .38). Because our
simulated correlations are uncorrected, the OBSERVED r is the correct comparison criterion. We recommend
re-anchoring targets on observed r before the camera-ready leaderboard.
