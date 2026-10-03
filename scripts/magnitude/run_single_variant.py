#!/usr/bin/env python3
"""Run ONE prompt variant on N personas (to remove the prompt-vs-sample-size confound by
producing a clean V0 baseline at the full N). Reuses magnitude_search machinery."""
import argparse, sys, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "magnitude"))
from magnitude_search import rate_variant, subset, Budget, VARIANTS, ModelSpec, LLMClient, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="V0_baseline")
ap.add_argument("--model", default="gpt-4.1", help="OpenAI rater model (A3 capability axis)")
ap.add_argument("--personas", type=int, default=300)
ap.add_argument("--days", type=int, default=6)
ap.add_argument("--cap", type=float, default=40.0)
a = ap.parse_args()

tag = a.model.upper().replace("-", "_").replace(".", "_")
spec = ModelSpec(name=tag, model_id=a.model, provider="openai")
client = LLMClient(spec)
b = Budget(a.cap)
emps = subset(a.personas)
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
vlabel = f"{a.variant}_{tag}"
log(f"single variant {a.variant} | model {a.model} | {len(emps)} personas")
out = rate_variant((vlabel, VARIANTS[a.variant]), client, spec, emps, a.days, b, log)
summ, df = evaluate(out)
df.to_csv(REPO / f"results/magnitude/pairs_{vlabel}.csv", index=False)
print(json.dumps(summ, indent=2))
log(f"DONE {vlabel} spent ${b.cost:.2f}")
