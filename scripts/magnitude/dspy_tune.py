#!/usr/bin/env python3
"""All-silico MIPROv2 achievability tune (appendix). Optimize the rating prompt against LITERATURE
targets only (norms.json state ICC/SD + per-persona seeded baselines/within-SDs + verified L3 coherence
pairs), held out by persona. Reports the optimizable internal-validity components (within-SD match, range use,
cross-state coherence) on a disjoint TEST split, baseline prompt vs MIPROv2-tuned. No real ESM.

Run: python scripts/magnitude/dspy_tune.py --model gpt-4.1-mini --auto light --train 30 --test 30
"""
import argparse, json, math, os, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
envf = REPO / ".env"
if envf.exists():
    for ln in envf.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
import dspy  # noqa: E402

SHARED = REPO / "results/generation/full_v2/_shared_events"
NORMS = json.loads((REPO / "data/seeds/norms.json").read_text())["states"]
SEED = json.loads((REPO / "data/seeds/synthetic_personas_seed.targets.json").read_text())
STATE_TGT = SEED["state_targets"]
PERSONA_TGT = {p["employee_id"]: p for p in SEED["personas"]}
XS = json.loads((REPO / "data/seeds/cross_state_targets.json").read_text())["pairs"]
VERIFIED = {k: v for k, v in XS.items() if not v.get("is_expert_prior", False)}  # tune only to verified
STATES = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]
BOUNDS = {s: tuple(NORMS[s]["scale"]) for s in STATES}


def scale_guidance():
    out = []
    for s in STATES:
        lo, hi = BOUNDS[s]
        out.append(f"{s.replace('_',' ').title()} ({lo}-{hi})")
    return "; ".join(out)


class RateDay(dspy.Signature):
    """You are the employee described. Report your honest end-of-day psychological states given who you
    are and what happened today. Let a genuinely strong day move your ratings toward the ends of each scale."""
    persona = dspy.InputField(desc="who this employee is (stable portrait)")
    narrative = dspy.InputField(desc="what happened to this employee today")
    scale_guidance = dspy.InputField(desc="state names and numeric scale bounds")
    work_engagement: float = dspy.OutputField(desc="1-7")
    psych_safety: float = dspy.OutputField(desc="1-4")
    work_creativity: float = dspy.OutputField(desc="1-6")
    team_trust: float = dspy.OutputField(desc="1-5")


class Rater(dspy.Module):
    def __init__(self):
        super().__init__(); self.rate = dspy.Predict(RateDay)

    def forward(self, persona, narratives, scale_guidance):
        series = {s: [] for s in STATES}
        for nar in narratives:
            try:
                r = self.rate(persona=persona, narrative=nar, scale_guidance=scale_guidance)
                for s in STATES:
                    lo, hi = BOUNDS[s]
                    series[s].append(max(lo, min(hi, float(getattr(r, s)))))
            except Exception:
                for s in STATES:
                    series[s].append((BOUNDS[s][0] + BOUNDS[s][1]) / 2)
        return dspy.Prediction(series=series)


def _within_corr(a, b):
    if len(a) < 3 or st.pstdev(a) == 0 or st.pstdev(b) == 0:
        return None
    ma, mb = st.mean(a), st.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    return cov / (st.pstdev(a) * st.pstdev(b))


def score(example, pred, trace=None):
    """Composite in [0,1]: within-SD match (.5) + range use (.2) + ceiling avoid (.1) + coherence (.2)."""
    series = pred.series
    sd_s, rng_s, ceil_s = [], [], []
    for s in STATES:
        v = series[s]; lo, hi = BOUNDS[s]
        tgt_sd = STATE_TGT[s]["within_sd_target"]
        rsd = st.pstdev(v) if len(v) > 1 else 0.0
        sd_s.append(math.exp(-((rsd - tgt_sd) / max(tgt_sd, 1e-6)) ** 2))      # gaussian around target
        rng_s.append(min(1.0, (max(v) - min(v)) / max(hi - lo, 1e-6) / 0.5))   # reward using >=half range
        pile = sum(1 for x in v if x <= lo + 0.05 * (hi - lo) or x >= hi - 0.05 * (hi - lo)) / len(v)
        ceil_s.append(1.0 - pile)
    coh = []
    for key, t in VERIFIED.items():
        a, b = key.split("__")
        r = _within_corr(series[a], series[b])
        if r is not None:
            coh.append(1.0 if 0.10 <= abs(r) <= 0.40 else max(0.0, 1 - abs(abs(r) - 0.25) / 0.25))
    coh_s = st.mean(coh) if coh else 0.5
    return 0.5 * st.mean(sd_s) + 0.2 * st.mean(rng_s) + 0.1 * st.mean(ceil_s) + 0.2 * coh_s


def build_examples(emps):
    ex = []
    for e in emps:
        ip = SHARED / f"{e}_identity.json"
        if not ip.exists():
            continue
        ident = json.loads(ip.read_text())
        nars = []
        for d in range(1, 7):
            dp = SHARED / f"{e}_day{d}_events.json"
            if dp.exists():
                nars.append(json.loads(dp.read_text())["narrative"])
        if len(nars) == 6:
            ex.append(dspy.Example(persona=ident["persona_prefix"], narratives=nars,
                                   scale_guidance=scale_guidance()).with_inputs("persona", "narratives", "scale_guidance"))
    return ex


def subset(n, seed=42, skip=0):
    import random
    e = sorted(p.name.replace("_identity.json", "") for p in SHARED.glob("*_identity.json"))
    random.Random(seed).shuffle(e)
    return sorted(e[skip:skip + n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    args = ap.parse_args()
    dspy.configure(lm=dspy.LM(f"openai/{args.model}", api_key=os.environ["OPENAI_API_KEY"],
                              max_tokens=400, temperature=0.8))
    train = build_examples(subset(args.train, skip=0))
    test = build_examples(subset(args.test, skip=150))
    print(f"[data] train={len(train)} test={len(test)} | verified L3 pairs={len(VERIFIED)} | model={args.model}")
    base = Rater()
    base_test = st.mean(score(e, base(**e.inputs())) for e in test)
    print(f"[baseline] held-out composite = {base_test:.3f}")
    # nohup-detached processes break DSPy's threaded async eval ('_ssock'); force single-thread + fresh loop.
    import asyncio
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
    except Exception:
        pass
    opt = dspy.MIPROv2(metric=score, auto=args.auto, num_threads=1)
    tuned = opt.compile(base, trainset=train, requires_permission_to_run=False)
    tuned_test = st.mean(score(e, tuned(**e.inputs())) for e in test)
    print(f"[tuned]    held-out composite = {tuned_test:.3f}  (delta {tuned_test-base_test:+.3f})")
    out = REPO / f"results/magnitude/dspy_tune_{args.model.replace('.','_').replace('-','_')}.json"
    out.write_text(json.dumps({"model": args.model, "auto": args.auto, "n_train": len(train),
                               "n_test": len(test), "baseline": base_test, "tuned": tuned_test}, indent=2))
    tuned.save(str(out.with_suffix(".prompt.json")))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
