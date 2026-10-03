# Cross-cultural validation results

## GEMMA_3_27B_IT
- **Measurement invariance**: mean |Δr| of per-culture cross-state correlations vs grand = 0.019 (lower = more invariant; ~<0.10 = strong metric invariance)
- **Cross-cultural convergence**: 50% of 4 Hofstede anchors match the published direction
  - PDI → psych_safety (expected -1, medium): r=-0.07 ✓
  - PDI → work_creativity (expected -1, low): r=-0.07 ✓
  - IDV → team_trust (expected -1, low): r=+0.28 ✗
  - IDV → work_engagement (expected +1, low): r=-0.06 ✗

  per-culture mean states:
  | culture | work_engagement | psych_safety | work_creativity | team_trust |
  |---|---|---|---|---|
  | Brazil | 3.47 | 2.30 | 2.99 | 2.71 |
  | China | 3.35 | 2.29 | 2.91 | 2.66 |
  | Germany | 3.42 | 2.31 | 2.98 | 2.68 |
  | India | 3.44 | 2.34 | 2.98 | 2.72 |
  | Japan | 3.42 | 2.31 | 2.91 | 2.64 |
  | USA | 3.38 | 2.31 | 2.94 | 2.70 |

## GEMMA_3_4B_IT
- **Measurement invariance**: mean |Δr| of per-culture cross-state correlations vs grand = 0.038 (lower = more invariant; ~<0.10 = strong metric invariance)
- **Cross-cultural convergence**: 75% of 4 Hofstede anchors match the published direction
  - PDI → psych_safety (expected -1, medium): r=-0.67 ✓
  - PDI → work_creativity (expected -1, low): r=-0.67 ✓
  - IDV → team_trust (expected -1, low): r=+0.69 ✗
  - IDV → work_engagement (expected +1, low): r=+0.56 ✓

  per-culture mean states:
  | culture | work_engagement | psych_safety | work_creativity | team_trust |
  |---|---|---|---|---|
  | Brazil | 3.42 | 2.40 | 3.02 | 3.12 |
  | China | 3.28 | 2.42 | 2.99 | 3.13 |
  | Germany | 3.38 | 2.52 | 3.05 | 3.14 |
  | India | 3.36 | 2.49 | 3.01 | 3.13 |
  | Japan | 3.35 | 2.50 | 3.00 | 3.08 |
  | USA | 3.39 | 2.48 | 3.01 | 3.20 |

## QWEN_2_5_72B_INSTRUCT
- **Measurement invariance**: mean |Δr| of per-culture cross-state correlations vs grand = 0.014 (lower = more invariant; ~<0.10 = strong metric invariance)
- **Cross-cultural convergence**: 75% of 4 Hofstede anchors match the published direction
  - PDI → psych_safety (expected -1, medium): r=-0.31 ✓
  - PDI → work_creativity (expected -1, low): r=-0.71 ✓
  - IDV → team_trust (expected -1, low): r=+0.46 ✗
  - IDV → work_engagement (expected +1, low): r=+0.64 ✓

  per-culture mean states:
  | culture | work_engagement | psych_safety | work_creativity | team_trust |
  |---|---|---|---|---|
  | Brazil | 3.79 | 2.48 | 2.98 | 2.78 |
  | China | 3.72 | 2.46 | 2.93 | 2.80 |
  | Germany | 3.81 | 2.49 | 3.05 | 2.80 |
  | India | 3.80 | 2.48 | 3.01 | 2.80 |
  | Japan | 3.75 | 2.48 | 3.00 | 2.74 |
  | USA | 3.79 | 2.47 | 3.00 | 2.82 |

