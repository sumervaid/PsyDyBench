#!/usr/bin/env python3
"""Pre-flight probe every roster model with one tiny call. Write a *_live.csv with only models that
respond, and report dead ones (404 no-endpoint, etc.). Prevents a dead model from aborting a run.
Usage: python3 preflight_roster.py data/seeds/model_roster_scaling.csv"""
import sys, csv, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import LLMClient
from gen_config import ModelSpec

REPO = Path(__file__).resolve().parents[2]
envf = REPO / ".env"
if envf.exists() and not os.environ.get("OPENROUTER_API_KEY"):
    for l in envf.read_text().splitlines():
        if l.strip() and not l.startswith("#") and "=" in l:
            k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

src = Path(sys.argv[1])
rows = list(csv.DictReader(open(src)))
live, dead = [], []
for r in rows:
    mid, prov = r["model_id"], r["provider"]
    if prov != "openrouter":          # openai models assumed live (billed separately)
        live.append(r); continue
    name = mid.split("/")[-1].upper().replace("-", "_").replace(".", "_")
    try:
        resp = LLMClient(ModelSpec(name, mid, prov)).chat("x", "reply OK", max_tokens=3, temperature=0.0)
        (live if (resp.text is not None) else dead).append(r)
        print(f"  LIVE {mid}", flush=True)
    except Exception as e:
        msg = str(e)[:70]; dead.append(r); print(f"  DEAD {mid}: {msg}", flush=True)
out = src.with_name(src.stem + "_live.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(live)
print(f"\nLIVE {len(live)} / DEAD {len(dead)} -> wrote {out}")
if dead: print("DEAD:", ", ".join(r["model_id"] for r in dead))
