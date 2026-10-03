#!/usr/bin/env python3
"""
Autonomous fix-search for the within-person MAGNITUDE problem.

Hypothesis: the production rating prompt (a) tells the rater its own trait baseline + typical
range (anchors -> regression to baseline -> compresses within-person movement) and (b) uses a
coarse integer scale. We test prompt interventions that should *de-anchor* and increase
within-person sensitivity, measuring within-person |r| (vs human meta 0.20-0.70) after each.

Self-improving: run base variants, rank, build a combo from the winners, validate the best
variant on the FULL set. Budget-capped (conservative meter). Resumable per variant.

Rater: gpt-4.1 (temperature controllable, cheap, deterministic-ish). Run ON the the compute host.
"""
import argparse, csv, json, os, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "generation"
sys.path.insert(0, str(GEN)); sys.path.insert(0, str(REPO / "scripts" / "magnitude"))

envf = REPO / ".env"
if envf.exists():
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from gen_config import ModelSpec, OUTCOMES, SCALE_INTERPRETATIONS   # noqa: E402
from llm_client import LLMClient                                    # noqa: E402
from stage3_rating import parse_states                             # noqa: E402
from eval_within_r import evaluate                                  # noqa: E402

SHARED = REPO / "results/generation/full_v2/_shared_events"
STATES = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]
SYSTEM = "You are the person described. Respond in first person, honestly."
RATES = (2.00, 0.50, 8.00)  # gpt-4.1 in/cached/out per 1M (conservative meter uses in,out)


def scale_guidance(ident, anchor: bool):
    lines = []
    for name, cfg in OUTCOMES.items():
        if anchor:
            rng = ident["observed_ranges"].get(name, [cfg.scale_min, cfg.scale_max])
            bl = ident["baselines"].get(name, (cfg.scale_min + cfg.scale_max) / 2)
            lines.append(f"\n{cfg.display_name} ({cfg.scale_min:.0f}-{cfg.scale_max:.0f}) "
                         f"- baseline {bl:.2f}, your typical range [{rng[0]:.1f}, {rng[1]:.1f}]:")
        else:
            lines.append(f"\n{cfg.display_name} ({cfg.scale_min:.0f}-{cfg.scale_max:.0f}):")
        lines.append(cfg.description)
        for val, desc in sorted(SCALE_INTERPRETATIONS.get(name, {}).items()):
            lines.append(f"  {val}: {desc}")
    return "\n".join(lines)


BASE_INSTR = """=== RATE YOUR END-OF-DAY STATE ===

You have just lived this day. It is 5:30pm. Report your ACTUAL internal state right
now - not how you should feel, how you do feel. Honor the texture of the day.

OUTPUT EXACTLY this block and nothing else (one number each, within the stated scale):

STATES:
- Work Engagement: [1-7] - [one sentence why]
- Psychological Safety: [1-4] - [one sentence why]
- Work Creativity: [1-6] - [one sentence why]
- Team Trust: [1-5] - [one sentence why]"""

DEVIATE = ("\n\nIMPORTANT: today is one specific day, not your average. If the day was genuinely "
           "good or bad, let your ratings move substantially toward the ends of each scale; do NOT "
           "default to a middle or 'typical' value.")
DECIMAL = "\n\nUse one decimal place (e.g., 5.3), reflecting how strongly today pushed each state."
SALIENT = ("\n\nFIRST, in one line, name the 2-3 specific things about TODAY that most moved your "
           "state (prefix 'DRIVERS:'). THEN output the STATES block.")

# variant -> (anchor_in_prefix, instruction_suffix)
VARIANTS = {
    "V0_baseline":  (True,  ""),
    "V1_noanchor":  (False, ""),
    "V2_deviate":   (False, DEVIATE),
    "V3_decimal":   (False, DEVIATE + DECIMAL),
    "V4_salient":   (False, DEVIATE + SALIENT),
}


def load_ident(emp):
    return json.loads((SHARED / f"{emp}_identity.json").read_text())


def load_day(emp, day):
    return json.loads((SHARED / f"{emp}_day{day}_events.json").read_text())


def subset(n, seed=42):
    emps = sorted(p.name.replace("_identity.json", "") for p in SHARED.glob("*_identity.json"))
    random.Random(seed).shuffle(emps)
    return sorted(emps[:n])


class Budget:
    def __init__(self, cap): self.cap = cap; self.cost = 0.0
    def add(self, it, ot): self.cost += it / 1e6 * RATES[0] + ot / 1e6 * RATES[2]
    def ok(self): return self.cost < self.cap


def rate_variant(vname, client, spec, emps, days, budget, log):
    anchor, suffix = VARIANTS[vname] if vname in VARIANTS else vname[1]
    if vname not in VARIANTS:  # combo passed as (name,(anchor,suffix))
        anchor, suffix = vname[1]; vname = vname[0]
    instr = BASE_INSTR + suffix
    out = REPO / f"results/magnitude/var_{vname}.csv"
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            done.add((r["employee_id"], r["day_index"]))
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["employee_id", "day_index", *STATES])
        for emp in emps:
            ident = load_ident(emp); sg = scale_guidance(ident, anchor)
            for day in range(1, days + 1):
                if (emp, str(day)) in done:
                    continue
                if not budget.ok():
                    log(f"[{vname}] BUDGET CAP hit at ${budget.cost:.2f}; stopping variant early");
                    return out
                d = load_day(emp, day)
                prefix = (f"{ident['persona_prefix']}\n=== TODAY (Day {d['day_index']}, "
                          f"{d['day_of_week']}) — DAY TYPE: {d['day_type']} ===\n\n{d['narrative']}"
                          f"\n\n=== SCALE GUIDANCE ===\n{sg}\n")
                spec.temperature_simulation = 0.8
                resp = client.chat(prefix, instr, max_tokens=spec.max_tokens_simulation,
                                   temperature=0.8)
                states, _, _ = parse_states(resp.text or "")
                budget.add(resp.input_tokens, resp.output_tokens)
                w.writerow([emp, day, *[states.get(s, "") for s in STATES]]); f.flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=int, default=50)
    ap.add_argument("--full", type=int, default=300, help="personas for final validation of winner")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--cap", type=float, default=150.0)
    args = ap.parse_args()

    logf = open(REPO / "results/magnitude/search.log", "a")
    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    spec = ModelSpec(name="GPT_4_1", model_id="gpt-4.1", provider="openai")
    client = LLMClient(spec)
    budget = Budget(args.cap)
    emps = subset(args.personas)
    log(f"=== magnitude search: {len(emps)} personas x {args.days} days, cap ${args.cap} ===")

    results = {}
    for v in VARIANTS:
        if not budget.ok():
            log("budget exhausted before " + v); break
        t0 = time.time()
        out = rate_variant(v, client, spec, emps, args.days, budget, log)
        summ, df = evaluate(out)
        df.to_csv(REPO / f"results/magnitude/pairs_{v}.csv", index=False)
        results[v] = summ
        log(f"[{v}] mean|within_r|={summ['mean_abs_within_r']:.3f} "
            f"(highconf {summ['mean_abs_within_r_highconf']:.3f}) "
            f"dir={summ['dirmatch_within_pct']:.0f}% rho={summ['rho_within_vs_meta']:.2f} "
            f"n={summ['n_pairs']} | {time.time()-t0:.0f}s | spent ${budget.cost:.2f}")

    # self-improve: combo = no-anchor + whichever of deviate/decimal/salient scored best on
    # mean|within_r| while keeping direction (dir>=55%)
    cand = {k: v for k, v in results.items()
            if k != "V0_baseline" and v["dirmatch_within_pct"] >= 55}
    combo_out = None
    if cand and budget.ok():
        best = max(cand, key=lambda k: cand[k]["mean_abs_within_r"])
        suffix = VARIANTS[best][1]
        combo = ("V5_combo", (False, suffix + DECIMAL if "DECIMAL" not in suffix else suffix))
        log(f"[self-improve] best base = {best}; building V5_combo")
        out = rate_variant(combo, client, spec, emps, args.days, budget, log)
        summ, df = evaluate(out); df.to_csv(REPO / "results/magnitude/pairs_V5_combo.csv", index=False)
        results["V5_combo"] = summ; combo_out = "V5_combo"
        log(f"[V5_combo] mean|within_r|={summ['mean_abs_within_r']:.3f} dir={summ['dirmatch_within_pct']:.0f}% "
            f"| spent ${budget.cost:.2f}")

    # validate the overall winner on the FULL persona set
    winner = max((k for k in results if k != "V0_baseline"),
                 key=lambda k: results[k]["mean_abs_within_r"], default=None)
    if winner and budget.ok() and args.full > args.personas:
        anchor, suffix = (VARIANTS[winner] if winner in VARIANTS
                          else (False, VARIANTS.get("V4_salient", (False, ""))[1]))
        log(f"[validate] winner={winner} on {args.full} personas")
        full_emps = subset(args.full)
        out = rate_variant((f"{winner}_full", (anchor, suffix)), client, spec, full_emps, args.days, budget, log)
        summ, df = evaluate(out); df.to_csv(REPO / f"results/magnitude/pairs_{winner}_full.csv", index=False)
        results[f"{winner}_full"] = summ
        log(f"[{winner}_full] mean|within_r|={summ['mean_abs_within_r']:.3f} "
            f"dir={summ['dirmatch_within_pct']:.0f}% n={summ['n_pairs']} | spent ${budget.cost:.2f}")

    # write summary
    with open(REPO / "results/magnitude/search_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + list(next(iter(results.values())).keys()))
        w.writeheader()
        for k, v in results.items():
            w.writerow({"variant": k, **v})
    log(f"=== DONE. total conservative spend ${budget.cost:.2f}. variants: "
        + ", ".join(f"{k}:{v['mean_abs_within_r']:.3f}" for k, v in results.items()) + " ===")


if __name__ == "__main__":
    main()
