# Magnitude experiment — results

Single-rating reliability rel_Y (gpt-4.1, temp 0.8): work_engagement=0.986, psych_safety=0.96, work_creativity=0.96, team_trust=0.973 (mean 0.97). High reliability => attenuation is NOT measurement noise.

## Prompt-intervention search (mean |within_r| vs human meta 0.20–0.70)

| variant | mean\|within_r\| | high-conf | dir-match % | rho vs meta | n_pairs |
|---|---|---|---|---|---|
| V0_baseline | 0.059 | 0.044 | 57.627 | 0.184 | 118 |
| V1_noanchor | 0.056 | 0.044 | 61.017 | 0.271 | 118 |
| V2_deviate | 0.059 | 0.044 | 58.475 | 0.197 | 118 |
| V3_decimal | 0.058 | 0.044 | 57.627 | 0.223 | 118 |
| V4_salient | 0.059 | 0.046 | 61.864 | 0.239 | 118 |
| V5_combo | 0.06 | 0.041 | 59.322 | 0.221 | 118 |
| V5_combo_full | 0.025 | 0.03 | 77.119 | 0.459 | 118 |

**Baseline** 0.059 -> **best (V5_combo)** 0.06 (+2%). Human meta band 0.20–0.70.

(Interpretation paragraph for the manuscript is written by hand from these numbers.)

## Track B — ICC-headroom (low-ICC states, gpt-4.1, de-anchored)

| state | ICC | n_pairs | mean\|within_r\| | dir% | rho |
|---|---|---|---|---|---|
| (trait-saturated 4) | ~0.75 | 118 | 0.023 | 71% | 0.38 |

Prediction under test: within|r| rises as 1-ICC rises. Affect (PA/NA halo) arm pending.

## Track B — ICC-headroom (low-ICC states, gpt-4.1, de-anchored)

| state | ICC | n_pairs | mean\|within_r\| | dir% | rho |
|---|---|---|---|---|---|
| daily_exhaustion_state | 0.34 | 8 | 0.058 | 100% | 0.69 |
| daily_fatigue_state | 0.4 | 8 | 0.075 | 100% | 0.95 |
| daily_flow_state | 0.48 | 8 | 0.069 | 75% | 0.53 |
| (trait-saturated 4) | ~0.75 | 118 | 0.023 | 71% | 0.38 |

Prediction under test: within|r| rises as 1-ICC rises. Affect (PA/NA halo) arm pending.
