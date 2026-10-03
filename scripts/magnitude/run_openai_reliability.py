#!/usr/bin/env python3
"""
Magnitude experiment — Stage 1: re-rate stored narratives with an OpenAI rater.

Goal: estimate the reliability of a single state rating (rel_Y) by re-rating the SAME
shared narratives multiple times, so we can disattenuate the within-person magnitude
(within-person |r| ~ 0.05 vs human 0.20-0.70). Indicators are IID-seeded (rel_X ~ 1),
so r_true = r_obs / sqrt(rel_Y).

Design (per rater):
  - gpt-4.1  : reps = 10x at temp 0.8  + 1x temp 0.0 + 1x temp 0.3   (reliability + temp sweep)
  - gpt-5.5  : reps = 8x at default temp (GPT-5 ignores temperature)  (reliability only)

Reuses the EXACT production rating path (scripts/generation/stage3_rating.rate_day) on the
once-generated narratives in results/generation/full_v2/_shared_events/.

Cost guard: conservative running cost (full input price, no cache discount) -> ABORT at --cap.
Resumable: skips (employee, day, rep) rows already in the output CSV.
NOTHING is regenerated; only the rating step is re-run. Run ON the the compute host (data + .env there).
"""
import argparse, csv, glob, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "generation"
sys.path.insert(0, str(GEN))

# load .env into os.environ (KEY=VALUE lines)
envf = REPO / ".env"
if envf.exists():
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from gen_config import ModelSpec          # noqa: E402
from llm_client import LLMClient          # noqa: E402
from stage3_rating import rate_day        # noqa: E402

SHARED = REPO / "results/generation/full_v2/_shared_events"
STATES = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]

# USD per 1M tokens: (input, cached_input, output). Conservative meter uses full input.
RATES = {"gpt-4.1": (2.00, 0.50, 8.00), "gpt-5.5": (5.00, 0.50, 30.00)}


def load_identity(emp):
    d = json.loads((SHARED / f"{emp}_identity.json").read_text())
    return SimpleNamespace(
        employee_id=emp,
        persona_prefix=d["persona_prefix"],
        observed_ranges=d.get("observed_ranges", {}),
        baselines=d.get("baselines", {}),
    )


def load_day(emp, day):
    d = json.loads((SHARED / f"{emp}_day{day}_events.json").read_text())
    return SimpleNamespace(
        employee_id=emp, day_index=int(d["day_index"]), day_of_week=d["day_of_week"],
        day_type=d["day_type"], narrative=d["narrative"], metadata={})


def rep_plan(model_id):
    """Return list of (rep_id, temperature) for one narrative."""
    if model_id.lower().startswith("gpt-5"):
        return [(f"r{i}_default", None) for i in range(8)]           # reliability only
    reps = [(f"r{i}_t0.8", 0.8) for i in range(10)]                  # reliability @ prod temp
    reps += [("t0.0", 0.0), ("t0.3", 0.3)]                          # temperature sweep
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gpt-4.1", "gpt-5.5"])
    ap.add_argument("--personas", type=int, default=50)
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--cap", type=float, default=80.0, help="hard USD cap (conservative); abort if exceeded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = args.model.upper().replace("-", "_").replace(".", "_")
    out = Path(args.out) if args.out else REPO / f"results/magnitude/ratings_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    # deterministic subset of personas that actually have identities
    emps = sorted(p.name.replace("_identity.json", "")
                  for p in SHARED.glob("*_identity.json"))
    random.Random(args.seed).shuffle(emps)
    emps = sorted(emps[:args.personas])
    print(f"[plan] model={args.model} personas={len(emps)} days={args.days} "
          f"reps/narr={len(rep_plan(args.model))} cap=${args.cap}")

    done = set()
    if out.exists():
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["employee_id"], row["day_index"], row["rep_id"]))
        print(f"[resume] {len(done)} rows already present")

    in_rate, _cached, out_rate = RATES[args.model]
    spec = ModelSpec(name=name, model_id=args.model, provider="openai")
    client = LLMClient(spec)

    header = ["employee_id", "day_index", "rep_id", "temperature", "rater",
              *STATES, "in_tok", "out_tok"]
    new = not out.exists()
    cost = 0.0
    n = 0
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for emp in emps:
            ident = load_identity(emp)
            for day in range(1, args.days + 1):
                d = load_day(emp, day)
                for rep_id, temp in rep_plan(args.model):
                    if (emp, str(day), rep_id) in done:
                        continue
                    spec.temperature_simulation = (temp if temp is not None else 1.0)
                    rd = rate_day(ident, d, client, spec)
                    it, ot = rd.metadata["tokens"]
                    cost += it / 1e6 * in_rate + ot / 1e6 * out_rate
                    w.writerow([emp, day, rep_id, ("default" if temp is None else temp),
                                args.model, *[rd.states.get(s, "") for s in STATES], it, ot])
                    f.flush()
                    n += 1
                    if n % 50 == 0:
                        print(f"[{n}] {emp} d{day} {rep_id}  running≈${cost:.2f}")
                    if cost > args.cap:
                        print(f"[ABORT] conservative cost ${cost:.2f} exceeded cap ${args.cap}. "
                              f"Wrote {n} new rows to {out}.")
                        return
    print(f"[done] {n} new calls, conservative cost ≈ ${cost:.2f} (real bill lower via caching). -> {out}")


if __name__ == "__main__":
    main()
