#!/usr/bin/env bash
# PsyDyBench: regenerate every leaderboard, figure, table, and reported number from the shipped corpora.
# This path runs offline: it reads only the generated ratings already in the repo.
#
#   python -m venv .venv && .venv/bin/pip install -r requirements.txt
#   PYTHON=.venv/bin/python ./reproduce.sh
#
# Stages run in dependency order. Each writes into results/. The final stage recomputes
# every macro in results/paper/values.tex and exits with status 1 on any disagreement.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
GEN=results/generation/full_v2
CULT=results/generation/cultural
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "integrity check of the generated corpora";            $PY scripts/analysis/divergence_check.py "$GEN"
log "internal validity, level pass/fail (leaderboard.csv)"; $PY scripts/validation/run_validation.py "$GEN"
log "internal validity, equivalence battery with persona-cluster bootstrap (leaderboard_v2.csv)"
                                                            $PY scripts/validation/run_validation_v2.py "$GEN" --n-boot 1000 --seed 17
log "external validity, indicator->state anchoring (anchoring_leaderboard.csv)"; $PY scripts/analysis/anchoring_leaderboard.py "$GEN"
log "external validity, trait->state nomological net (nomological_leaderboard.csv)"; $PY scripts/analysis/nomological_scaling.py "$GEN"
log "scaling slopes vs parameter count (scaling_slopes.csv)"; $PY scripts/analysis/scaling_curves.py
log "re-anchoring on full-text-verified targets (reanchored_convergence.csv)"; $PY scripts/analysis/reanchor_convergence.py "$GEN"
log "threshold sensitivity of the level verdicts (threshold_sensitivity.csv)"; $PY scripts/analysis/threshold_sensitivity.py
log "cross-cultural invariance (cultural_validation.csv)";   $PY scripts/validation/validate_cultural.py "$CULT"
# The appendix magnitude probe (results/magnitude/) is scored directly by verify_paper_values.py
# from the shipped rating files. Its generation scripts call paid APIs and are documented in docs/METHOD.md.
log "figures (results/figures/fig_*.pdf)";                   $PY scripts/figures/simbench_style_figures.py
log "appendix leaderboard table (results/paper/model_ranking_table.tex)"; $PY scripts/magnitude/gen_ranking_table.py
log "verify every number in the paper against the regenerated files"; $PY scripts/verify_paper_values.py --write-doc
log "done"
