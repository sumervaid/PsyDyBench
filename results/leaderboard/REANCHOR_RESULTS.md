# Re-anchored convergence: observed r vs claimed targets

18 of 18 verified pairs overrode their META_BENCHMARKS target with full-text OBSERVED r (the rest of the ~118-pair library unchanged). Models scored: 29 (preliminary — full 48-model run still finishing).

Mean change in pooled direction-match%: +0.00 pts (per model).

## Per-model: original (claimed targets) vs re-anchored (observed r)

| model | orig dir% | reanch dir% | orig rho | reanch rho |
|---|---|---|---|---|
| LLAMA_3_2_3B_INSTRUCT | 81.4 | 81.4 | 0.521 | 0.503 |
| GRANITE_4_1_8B | 77.1 | 77.1 | 0.569 | 0.557 |
| GEMMA_3_4B_IT | 74.6 | 74.6 | 0.567 | 0.57 |
| QWEN3_30B_A3B_THINKING_2507 | 74.6 | 74.6 | 0.396 | 0.374 |
| GEMMA_3_12B_IT | 73.7 | 73.7 | 0.536 | 0.531 |
| GPT_5_5 | 73.7 | 73.7 | 0.464 | 0.446 |
| QWEN_2_5_7B_INSTRUCT | 73.7 | 73.7 | 0.42 | 0.416 |
| LLAMA_4_MAVERICK | 72.9 | 72.9 | 0.442 | 0.435 |
| QWEN_2_5_72B_INSTRUCT | 72.9 | 72.9 | 0.479 | 0.457 |
| DEEPSEEK_R1_0528 | 72.0 | 72.0 | 0.548 | 0.538 |
| QWEN3_30B_A3B_INSTRUCT_2507 | 71.2 | 71.2 | 0.433 | 0.414 |
| GEMMA_2_27B_IT | 70.3 | 70.3 | 0.441 | 0.429 |
| GEMMA_3_27B_IT | 70.3 | 70.3 | 0.452 | 0.46 |
| GPT_OSS_20B | 70.3 | 70.3 | 0.391 | 0.382 |
| LLAMA_3_1_8B_INSTRUCT | 70.3 | 70.3 | 0.434 | 0.43 |
| LLAMA_3_3_70B_INSTRUCT | 70.3 | 70.3 | 0.411 | 0.395 |
| QWEN3_235B_A22B_2507 | 70.3 | 70.3 | 0.392 | 0.363 |
| QWEN3_8B | 70.3 | 70.3 | 0.351 | 0.337 |
| GPT_OSS_120B | 69.5 | 69.5 | 0.393 | 0.379 |
| PHI_4 | 69.5 | 69.5 | 0.422 | 0.427 |
| QWEN3_32B | 69.5 | 69.5 | 0.38 | 0.361 |
| CLAUDE_SONNET_4 | 68.6 | 68.6 | 0.437 | 0.43 |
| GPT_4_1 | 68.6 | 68.6 | 0.309 | 0.3 |
| QWEN3_14B | 68.6 | 68.6 | 0.368 | 0.346 |
| DEEPSEEK_V3_2 | 66.1 | 66.1 | 0.426 | 0.408 |
| LLAMA_3_2_1B_INSTRUCT | 66.1 | 66.1 | 0.34 | 0.347 |
| MISTRAL_SMALL_3_2_24B_INSTRUCT | 66.1 | 66.1 | 0.421 | 0.413 |
| GEMINI_2_5_PRO | 63.6 | 63.6 | 0.355 | 0.34 |
| MISTRAL_SMALL_24B_INSTRUCT_2501 | None | None | None | None |

## Per-pair: simulated r vs claimed target vs verified observed r
(pooled/between-person simulated r, averaged across models; the 21 verified pairs)

| pair | claimed target | verified obs r | mean simulated r |
|---|---|---|---|
| psychological_autonomy_felt -> psych_safety_state | 0.47 | 0.38 | 0.004 |
| psychological_competence_affirmation -> psych_safety_state | 0.4 | 0.48 | 0.018 |
| psychological_meaning_connection -> psych_safety_state | 0.35 | 0.41 | -0.01 |
| social_support_received_count -> psych_safety_state | 0.49 | 0.4 | 0.036 |
| social_peer_interaction_count -> psych_safety_state | 0.3 | 0.5 | 0.001 |
| social_support_given_count -> psych_safety_state | 0.25 | 0.4 | -0.003 |
| psychological_autonomy_felt -> work_engagement_state | 0.39 | 0.31 | 0.007 |
| psychological_anticipatory_excitement -> work_engagement_state | 0.35 | 0.3 | 0.013 |
| psychological_anticipatory_dread -> work_engagement_state | -0.25 | -0.16 | -0.004 |
| psychological_stress_level -> work_engagement_state | -0.19 | -0.2 | -0.049 |
| social_conflict_instances -> work_engagement_state | -0.19 | -0.2 | -0.033 |
| social_manager_interaction_quality -> team_trust_state | 0.46 | 0.72 | 0.026 |
| social_support_received_count -> team_trust_state | 0.4 | 0.69 | 0.051 |
| psychological_competence_affirmation -> team_trust_state | 0.35 | 0.55 | 0.011 |
| behavioral_overtime_indication -> work_engagement_state | -0.15 | -0.18 | -0.005 |
| life_work_personal_stress_level -> work_engagement_state | -0.22 | -0.3 | 0.01 |
| psychological_affect_arousal -> work_creativity_state | 0.2 | 0.19 | 0.007 |
| psychological_competence_affirmation -> work_creativity_state | 0.3 | 0.13 | 0.025 |

**Reading:** if direction-match barely moves, the convergence finding is robust to the corrected-vs-observed target choice (the SIGNS agree even where magnitudes differ). Magnitude differences show up in the per-pair table (claimed=corrected rho tends to exceed observed r).
