#!/usr/bin/env bash
# PsyDyBench: run the generation + rating pipeline. THIS STAGE CALLS PAID APIs.
# Everything downstream (reproduce.sh) reads the shipped ratings and is free. This script is
# needed only to add a model or regenerate from scratch.
#
# Keys are read from the environment and kept out of every file:
#   export OPENROUTER_API_KEY=...   # open-weight models
#   export OPENAI_API_KEY=...       # OpenAI models
#
# Dry run, offline, on a stub client (exercises the whole pipeline):
#   ./scripts/run_benchmark.sh --mock --limit 3
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
OUT="${OUT:-results/generation/full_v2}"
ROSTER="${ROSTER:-data/seeds/model_roster_scaling_live.csv}"
# Budget guardrails (USD) at which the orchestrator halts. Raise deliberately.
OR_LIMIT="${OR_LIMIT:-50}"; OAI_LIMIT="${OAI_LIMIT:-50}"

$PY scripts/generation/run_benchmark.py \
    --personas data/seeds/personas_full.csv --out "$OUT" --roster "$ROSTER" \
    --n-days 6 --workers "${WORKERS:-16}" --or-limit "$OR_LIMIT" --oai-limit "$OAI_LIMIT" "$@"
