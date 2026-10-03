#!/usr/bin/env python3
"""ICC-headroom experiment: rate 3 low-ICC states (exhaustion/fatigue/flow) from the SAME shared
narratives, then test whether synthetic within-person |r| rises with 1-ICC vs the trait-saturated
four states (within|r|~0.023 at ICC~0.7-0.8). De-anchored population-level scale guidance (no
person baseline), gpt-4.1. Re-rate only; no regeneration. Run on the compute host."""
import csv, json, os, re, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "generation"))
sys.path.insert(0, str(REPO / "scripts" / "magnitude"))
envf = REPO / ".env"
if envf.exists():
    for ln in envf.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
from gen_config import ModelSpec                       # noqa: E402
from llm_client import LLMClient                        # noqa: E402
from eval_within_r import compute_within_r              # noqa: E402 (stubs applied there)
from new_states import NEW_STATES, NEW_STATE_BENCHMARKS  # noqa: E402
import pandas as pd, numpy as np                        # noqa: E402

SHARED = REPO / "results/generation/full_v2/_shared_events"
COLS = list(NEW_STATES.keys())
SYS = "You are the person described. Respond in first person, honestly."
PAT = {"daily_exhaustion_state": r"exhaustion[:\s]*\[?(\d+\.?\d*)",
       "daily_fatigue_state": r"fatigue[:\s]*\[?(\d+\.?\d*)",
       "daily_flow_state": r"(?:flow|absorption)[:\s]*\[?(\d+\.?\d*)"}


def guidance():
    out = []
    for col, c in NEW_STATES.items():
        out.append(f"\n{c['display']} ({c['min']}-{c['max']}):")
        for v, d in sorted(c["interp"].items()):
            out.append(f"  {v}: {d}")
    return "\n".join(out)

INSTR = ("=== RATE YOUR END-OF-DAY STATE ===\nIt is 5:30pm; report your ACTUAL state now, honoring the "
         "texture of today. Let a genuinely good or bad day move you toward the ends of each scale.\n\n"
         "OUTPUT EXACTLY:\nSTATES:\n- Emotional Exhaustion: [1-7] - [why]\n- Fatigue: [1-5] - [why]\n"
         "- Flow/Absorption: [1-7] - [why]")


def parse(text):
    out = {}
    for col, pat in PAT.items():
        m = re.search(pat, text.lower())
        if m:
            c = NEW_STATES[col]
            out[col] = max(c["min"], min(c["max"], float(m.group(1))))
    return out


def subset(n, seed=42):
    import random
    e = sorted(p.name.replace("_identity.json", "") for p in SHARED.glob("*_identity.json"))
    random.Random(seed).shuffle(e); return sorted(e[:n])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--personas", type=int, default=300)
    ap.add_argument("--days", type=int, default=6); ap.add_argument("--cap", type=float, default=30)
    a = ap.parse_args()
    spec = ModelSpec(name="GPT_4_1_NEWSTATES", model_id="gpt-4.1", provider="openai")
    client = LLMClient(spec)
    out = REPO / "results/magnitude/ratings_newstates.csv"
    g = guidance()
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            done.add((r["employee_id"], r["day_index"]))
    emps = subset(a.personas)
    cost = 0.0
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["employee_id", "day_index", *COLS])
        for emp in emps:
            ident = json.loads((SHARED / f"{emp}_identity.json").read_text())
            for day in range(1, a.days + 1):
                if (emp, str(day)) in done:
                    continue
                d = json.loads((SHARED / f"{emp}_day{day}_events.json").read_text())
                prefix = (f"{ident['persona_prefix']}\n=== TODAY (Day {d['day_index']}, {d['day_of_week']})"
                          f" — DAY TYPE: {d['day_type']} ===\n\n{d['narrative']}\n\n=== SCALE GUIDANCE ===\n{g}\n")
                resp = client.chat(prefix, INSTR, max_tokens=spec.max_tokens_simulation, temperature=0.8)
                st = parse(resp.text or "")
                cost += resp.input_tokens / 1e6 * 2.0 + resp.output_tokens / 1e6 * 8.0
                w.writerow([emp, day, *[st.get(c, "") for c in COLS]]); f.flush()
                if cost > a.cap:
                    print(f"[ABORT] cap ${a.cap} hit at ${cost:.2f}"); return
    print(f"[rated] conservative ${cost:.2f}")
    evaluate()


def evaluate():
    s = pd.read_csv(REPO / "results/magnitude/ratings_newstates.csv").rename(columns={"day_index": "day"})
    s = s.groupby(["employee_id", "day"])[COLS].mean().reset_index()
    ind = pd.read_csv(REPO / "data/seeds/personas_full.csv")
    ind = ind.drop(columns=[c for c in COLS if c in ind.columns], errors="ignore")
    m = ind.merge(s, on=["employee_id", "day"], how="inner")
    rows = []
    for e in NEW_STATE_BENCHMARKS:
        i, st, mr = e["indicator"], e["state"], e["meta_r"]
        if i in m.columns and st in m.columns:
            rows.append({"state": st, "indicator": i, "meta_r": mr,
                         "within_r": compute_within_r(m, i, st), "confidence": e["confidence"]})
    df = pd.DataFrame(rows); df.to_csv(REPO / "results/magnitude/newstates_within_r.csv", index=False)
    print("\nstate | ICC | n | mean|within_r| | dir% | rho_vs_meta")
    summary = []
    for st in COLS:
        d = df[(df.state == st)].dropna(subset=["within_r"])
        if not len(d):
            continue
        mabs = d["within_r"].abs().mean()
        dirp = 100 * (np.sign(d["within_r"]) == np.sign(d["meta_r"])).mean()
        rho = d["within_r"].corr(d["meta_r"], method="spearman") if len(d) > 2 else float("nan")
        icc = NEW_STATES[st]["icc"]
        summary.append((st, icc, mabs, dirp, rho, len(d)))
        print(f"{st} | {icc} | {len(d)} | {mabs:.3f} | {dirp:.0f}% | {rho:.2f}")
    # reference: trait-saturated 4 states sit at within|r|~0.023, ICC~0.75
    print("\n(reference: trait-saturated 4 states  ICC~0.75  within|r|~0.023)")
    with open(REPO / "results/magnitude/MAGNITUDE_RESULTS.md", "a") as f:
        f.write("\n## Track B — ICC-headroom (low-ICC states, gpt-4.1, de-anchored)\n\n")
        f.write("| state | ICC | n_pairs | mean\\|within_r\\| | dir% | rho |\n|---|---|---|---|---|---|\n")
        for st, icc, mabs, dirp, rho, n in summary:
            f.write(f"| {st} | {icc} | {n} | {mabs:.3f} | {dirp:.0f}% | {rho:.2f} |\n")
        f.write("| (trait-saturated 4) | ~0.75 | 118 | 0.023 | 71% | 0.38 |\n")
        f.write("\nPrediction under test: within|r| rises as 1-ICC rises. Affect (PA/NA halo) arm pending.\n")


if __name__ == "__main__":
    main()
