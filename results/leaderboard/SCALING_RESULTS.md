# Scaling-curve analysis — longitudinal-persona fidelity vs model size

Fit: metric ~ log10(params_b). Dense models only in the fit; MoE shown by total+active separately; closed/reference models excluded (unknown params). Run after the scaling benchmark; metrics for not-yet-run models are blank.

## var_fidelity  (lower=better)
- **pooled (all dense, n=21)**: slope=-0.201 per decade, r=-0.14
  - Gemma-3 (n=3): slope=-0.445, r=-0.98
  - Llama-3.x (n=4): slope=+0.034, r=+0.28
  - Qwen2.5 (n=2): slope=-0.177, r=-1.00
  - Qwen3 (n=3): slope=+0.062, r=+0.49

## internal_validity  (higher=better)
- **pooled (all dense, n=22)**: slope=-0.064 per decade, r=-0.29
  - Gemma-3 (n=3): slope=-0.156, r=-0.84
  - Llama-3.x (n=4): slope=-0.113, r=-1.00
  - Qwen2.5 (n=2): slope=+0.006, r=+1.00
  - Qwen3 (n=3): slope=-0.072, r=-0.99

## anchor_pooled_dir%  (higher=better)
- **pooled (all dense, n=21)**: slope=-0.665 per decade, r=-0.04
  - Gemma-3 (n=3): slope=-4.996, r=-0.92
  - Llama-3.x (n=4): slope=-0.145, r=-0.02
  - Qwen2.5 (n=2): slope=-0.790, r=-1.00
  - Qwen3 (n=3): slope=-1.039, r=-0.37

## anchor_pooled_rho  (higher=better)
- **pooled (all dense, n=21)**: slope=+0.022 per decade, r=+0.07
  - Gemma-3 (n=3): slope=-0.134, r=-0.94
  - Llama-3.x (n=4): slope=+0.013, r=+0.13
  - Qwen2.5 (n=2): slope=+0.058, r=+1.00
  - Qwen3 (n=3): slope=+0.047, r=+0.98

## anchor_within_dir%  (higher=better)
- **pooled (all dense, n=21)**: slope=+1.993 per decade, r=+0.11
  - Gemma-3 (n=3): slope=+0.709, r=+0.09
  - Llama-3.x (n=4): slope=+0.414, r=+0.11
  - Qwen2.5 (n=2): slope=+0.790, r=+1.00
  - Qwen3 (n=3): slope=+10.849, r=+0.75

## Interpretation hook
Compare slope SIGNS across metrics: if variance-fidelity and anchor-convergence scale in OPPOSITE directions, longitudinal-persona validity does not follow a single capability scaling law — the central finding. [[v2: fill once the scaling benchmark completes]]
