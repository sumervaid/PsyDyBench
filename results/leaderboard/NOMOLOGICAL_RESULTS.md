# Nomological external validity (trait->state, BETWEEN-person) + known-groups

**The clean external check.** Big Five traits are seeded with real between-person variance, so trait->state effects are recoverable against published targets (unlike IID daily indicators). This is estimand-matched construct validity.

Models scored: 34.  Mean nomological direction-match: **84.3%** (chance 50%), range [75, 100].

## Leaderboard (trait->state direction agreement)

| model | nomo dir% | nomo rho | n pairs | params(B) |
|---|---|---|---|---|
| GEMMA_3_4B_IT | 100.0 | 0.835 | 20 | 4 |
| DEEPSEEK_R1_DISTILL_LLAMA_70B | 90.0 | 0.744 | 20 | 70 |
| GEMMA_3_12B_IT | 90.0 | 0.7 | 20 | 12 |
| GEMMA_3_27B_IT | 90.0 | 0.843 | 20 | 27 |
| GRANITE_4_1_8B | 90.0 | 0.721 | 20 | 8 |
| QWEN3_235B_A22B_2507 | 90.0 | 0.684 | 20 | 235 |
| QWEN3_30B_A3B_THINKING_2507 | 90.0 | 0.637 | 20 | 30 |
| QWEN3_8B | 90.0 | 0.568 | 20 | 8 |
| QWEN_2_5_72B_INSTRUCT | 90.0 | 0.802 | 20 | 72 |
| DEEPSEEK_R1_0528 | 85.0 | 0.706 | 20 | 671 |
| DEEPSEEK_V3_2 | 85.0 | 0.788 | 20 | 671 |
| GEMINI_2_5_FLASH | 85.0 | 0.779 | 20 |  |
| GEMINI_2_5_PRO | 85.0 | 0.751 | 20 |  |
| GPT_4_1 | 85.0 | 0.627 | 20 |  |
| GPT_OSS_120B | 85.0 | 0.774 | 20 | 117 |
| LLAMA_3_1_8B_INSTRUCT | 85.0 | 0.651 | 20 | 8 |
| LLAMA_3_8B_INSTRUCT | 85.0 | 0.581 | 20 | 8 |
| PHI_4 | 85.0 | 0.589 | 20 | 14 |
| QWEN3_30B_A3B_INSTRUCT_2507 | 85.0 | 0.675 | 20 | 30 |
| QWEN3_32B | 85.0 | 0.718 | 20 | 32 |
| QWEN_2_5_7B_INSTRUCT | 85.0 | 0.715 | 20 | 7 |
| CLAUDE_SONNET_4 | 80.0 | 0.629 | 20 |  |
| GEMMA_2_27B_IT | 80.0 | 0.806 | 20 | 27 |
| GPT_5_5 | 80.0 | 0.694 | 20 |  |
| LLAMA_3_1_70B_INSTRUCT | 80.0 | 0.598 | 20 | 70 |
| LLAMA_3_2_3B_INSTRUCT | 80.0 | 0.61 | 20 | 3 |
| LLAMA_3_3_70B_INSTRUCT | 80.0 | 0.547 | 20 | 70 |
| LLAMA_4_MAVERICK | 80.0 | 0.715 | 20 | 400 |
| QWEN3_14B | 80.0 | 0.676 | 20 | 14 |
| QWEN3_5_9B | 80.0 | 0.53 | 20 | 9 |
| QWEN_2_5_CODER_32B_INSTRUCT | 80.0 | 0.653 | 20 | 32 |
| GPT_OSS_20B | 75.0 | 0.776 | 20 | 21 |
| LLAMA_3_2_1B_INSTRUCT | 75.0 | 0.496 | 20 | 1 |
| MISTRAL_SMALL_3_2_24B_INSTRUCT | 75.0 | 0.663 | 20 | 24 |

## Scaling tie-in
Dense-model fit: nomological direction% ~ log10(params): slope=-0.67 pts/decade, r=-0.052 (n=20). [[PENDING: refit on full 48-model roster.]]
This is the FOURTH validity dimension for the scaling analysis (alongside variance fidelity, the internal-validity battery, indicator convergence) — does EXTERNAL nomological validity track size?

## Known-groups validity (high vs low Neuroticism)
Expectation: higher Neuroticism -> lower engagement/safety MEAN and higher day-to-day VOLATILITY.

| model | Δengagement mean | Δpsych_safety mean | Δengagement volatility |
|---|---|---|---|
| CLAUDE_SONNET_4 | -0.104 | -0.065 | 0.02 |
| DEEPSEEK_R1_0528 | -0.086 | -0.1 | 0.01 |
| DEEPSEEK_R1_DISTILL_LLAMA_70B | -0.07 | -0.111 | 0.015 |
| DEEPSEEK_V3_2 | -0.095 | -0.078 | 0.039 |
| GEMINI_2_5_FLASH | -0.127 | -0.123 | 0.035 |
| GEMINI_2_5_PRO | -0.071 | -0.058 | 0.032 |
| GEMMA_2_27B_IT | -0.116 | -0.151 | -0.011 |
| GEMMA_3_12B_IT | -0.135 | -0.13 | -0.045 |
| GEMMA_3_27B_IT | -0.131 | -0.15 | -0.057 |
| GEMMA_3_4B_IT | -0.113 | -0.144 | -0.053 |
| GPT_4_1 | -0.04 | -0.137 | 0.034 |
| GPT_5_5 | -0.044 | -0.122 | 0.004 |
| GPT_OSS_120B | -0.102 | -0.17 | 0.051 |
| GPT_OSS_20B | -0.109 | -0.16 | 0.061 |
| GRANITE_4_1_8B | -0.155 | -0.14 | 0.042 |
| LLAMA_3_1_70B_INSTRUCT | -0.13 | -0.146 | 0.041 |
| LLAMA_3_1_8B_INSTRUCT | -0.12 | -0.171 | 0.006 |
| LLAMA_3_2_1B_INSTRUCT | -0.076 | -0.035 | -0.071 |
| LLAMA_3_2_3B_INSTRUCT | -0.083 | -0.189 | -0.001 |
| LLAMA_3_3_70B_INSTRUCT | -0.101 | -0.113 | 0.052 |
| LLAMA_3_8B_INSTRUCT | -0.12 | -0.19 | 0.024 |
| LLAMA_4_MAVERICK | -0.069 | -0.129 | 0.034 |
| MISTRAL_SMALL_3_2_24B_INSTRUCT | -0.107 | -0.146 | 0.057 |
| PHI_4 | -0.139 | -0.145 | -0.0 |
| QWEN3_14B | -0.045 | -0.128 | 0.025 |
| QWEN3_235B_A22B_2507 | -0.06 | -0.152 | 0.014 |
| QWEN3_30B_A3B_INSTRUCT_2507 | -0.068 | -0.175 | 0.034 |
| QWEN3_30B_A3B_THINKING_2507 | -0.098 | -0.158 | 0.015 |
| QWEN3_32B | -0.078 | -0.107 | -0.016 |
| QWEN3_5_9B | -0.01 | -0.004 | 0.04 |
| QWEN3_8B | -0.044 | -0.126 | 0.006 |
| QWEN_2_5_72B_INSTRUCT | -0.132 | -0.142 | 0.035 |
| QWEN_2_5_7B_INSTRUCT | -0.054 | -0.134 | 0.004 |
| QWEN_2_5_CODER_32B_INSTRUCT | -0.098 | -0.137 | 0.049 |
