# Threshold-sensitivity analysis (the internal-validity battery (v2))

**Claim 1 — the leaderboard ordering is threshold-free.** Models are ranked by continuous distances (mean |ln within-SD ratio|, mean |Δr|), which contain no margin. Perturbing margins cannot change this ranking (Kendall tau = 1.000 by construction).

**Claim 2 — the secondary equivalence-count ranking is stable.** Re-evaluating every criterion's verdict from its stored bootstrap CI at perturbed margins and recomputing the count-ranking:

| margin factor | Kendall tau vs baseline (factor 1.0) |
|---|---|
| 0.50x | +1.000 |
| 0.80x | +1.000 |
| 1.00x | +1.000 |
| 1.25x | +1.000 |
| 2.00x | +1.000 |

Kendall tau near +1 across the 0.5x-2.0x range = the equivalence-count ranking is robust to threshold choice. The continuous ranking is threshold-free by construction.
