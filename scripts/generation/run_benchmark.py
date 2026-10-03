#!/usr/bin/env python3
"""
run_benchmark.py — hardened, CONCURRENT, RESUMABLE, BUDGET-GUARDED benchmark orchestrator.

Fixes over run_generation.py (serial, ~100h for the full run):
  - ThreadPool concurrency (--workers) over the finest unit of work.
  - Resumable: skips any unit whose output already exists; on restart it RELOADS the
    persisted shared narratives (never regenerates — keeps shared-event consistency).
  - Per-call usage logging -> results/budget/{openai,openrouter}_usage.jsonl (budget tracker).
  - Budget guardrail: polls spend; HALTS if OpenRouter or OpenAI exceeds --budget-limit ($50).
  - Roster read from data/seeds/model_roster.csv (pre-registered); provider per row.

Two phases:
  1) GEN  (shared events, fixed event model): identity + N day narratives per persona, persisted.
  2) RATE (per roster model): rate the SAME persisted narratives. Flat-parallel over (persona,day,model).
"""
import os, sys, csv, json, time, random, argparse, threading, urllib.request
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_config import RunConfig, ModelSpec, OUTCOMES
from llm_client import make_client
from stage1_identity import generate_identity, Identity
from stage2_events import plan_variance_coverage, narrativize_day, NarrativizedDay
from stage3_rating import rate_day

REPO = Path(__file__).resolve().parents[2]
BUD = REPO / "results/budget"; BUD.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()

# ---- usage logging (thread-safe) -------------------------------------------------
def log_usage(provider, model, tin, tout):
    fn = BUD / ("openai_usage.jsonl" if provider == "openai" else "openrouter_usage.jsonl")
    with _LOCK:
        with open(fn, "a") as f:
            f.write(json.dumps({"model": model, "prompt_tokens": int(tin or 0),
                                "completion_tokens": int(tout or 0)}) + "\n")

# ---- budget polling (OpenRouter ground-truth + OpenAI token estimate) -------------
OAI_PRICES = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
              "gpt-5.5": (2.50, 15.0), "gpt-5.4": (2.0, 12.0), "gpt-5.1": (1.25, 10.0), "_d": (3.0, 15.0)}

def openrouter_spend():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: return 0.0
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                     headers={"Authorization": f"Bearer {key}"})
        d = json.load(urllib.request.urlopen(req, timeout=15)); dd = d.get("data", d)
        base = 0.0
        bf = BUD / "openrouter_baseline.json"
        if bf.exists(): base = json.loads(bf.read_text())["openrouter_baseline_usage"]
        return float(dd.get("total_usage", 0.0)) - base
    except Exception:
        return 0.0

def openai_spend():
    f = BUD / "openai_usage.jsonl"
    if not f.exists(): return 0.0
    tot = 0.0
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); pin, pout = OAI_PRICES.get(r["model"], OAI_PRICES["_d"])
        tot += r["prompt_tokens"] * pin / 1e6 + r["completion_tokens"] * pout / 1e6
    return tot

def budget_ok(or_limit, oai_limit):
    o, a = openrouter_spend(), openai_spend()
    return (o <= or_limit and a <= oai_limit), o, a

# ---- reconstruct shared objects from disk (resume-safe) --------------------------
def load_identity(eid, ndir):
    d = json.loads((ndir / f"{eid}_identity.json").read_text())
    return Identity(employee_id=eid, first_name="", last_name="", portrait=d.get("portrait", ""),
                    reactivity_patterns="", observed_ranges=d.get("observed_ranges", {}),
                    baselines=d.get("baselines", {}), within_sds={},
                    persona_prefix=d["persona_prefix"], metadata=d.get("metadata", {}))

def load_day(eid, di, ndir):
    d = json.loads((ndir / f"{eid}_day{di}_events.json").read_text())
    return NarrativizedDay(employee_id=eid, day_index=d["day_index"], day_of_week=d["day_of_week"],
                           day_type=d["day_type"], narrative=d["narrative"], metadata=d.get("metadata", {}))

# ---- work units -----------------------------------------------------------------
def gen_persona(persona, ev_client, ev_spec, rc, ndir, indicator_lookup=None):
    eid = str(persona["employee_id"])
    rng = random.Random((hash((rc.seed, eid))) & 0xffffffff)
    day_types = plan_variance_coverage(rc.n_days_per_persona, rc, rng)
    idp = ndir / f"{eid}_identity.json"
    if not (idp.exists() and idp.stat().st_size > 0):
        idn = generate_identity(persona, ev_client, ev_spec)
        log_usage(ev_spec.provider, ev_spec.model_id, *idn.metadata["tokens"])
        idp.write_text(json.dumps({"portrait": idn.portrait, "persona_prefix": idn.persona_prefix,
                                   "baselines": idn.baselines, "observed_ranges": idn.observed_ranges,
                                   "metadata": idn.metadata}, indent=2))
    identity = load_identity(eid, ndir)
    for di, dt in enumerate(day_types, start=1):
        dp = ndir / f"{eid}_day{di}_events.json"
        if dp.exists() and dp.stat().st_size > 0: continue
        inds = (indicator_lookup or {}).get((eid, di))
        nd = narrativize_day(identity, di, dt, ev_client, ev_spec, indicators=inds)
        log_usage(ev_spec.provider, ev_spec.model_id, *nd.metadata["tokens"])
        dp.write_text(json.dumps({"day_index": di, "day_of_week": nd.day_of_week, "day_type": dt,
                                  "narrative": nd.narrative, "metadata": nd.metadata}, indent=2))
    return eid

def rate_unit(eid, di, spec, client, ndir, out_dir):
    op = out_dir / spec.name / f"{eid}_day{di}.json"
    if op.exists() and op.stat().st_size > 0: return "skip"
    identity, day = load_identity(eid, ndir), load_day(eid, di, ndir)
    rated = rate_day(identity, day, client, spec)
    log_usage(spec.provider, spec.model_id, *rated.metadata["tokens"])
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps({"employee_id": eid, "day_index": di, "day_of_week": rated.day_of_week,
                              "day_type": rated.day_type, "states": rated.states,
                              "state_reasoning": rated.state_reasoning, "metadata": rated.metadata}, indent=2))
    return "done"

def pool_run(fn, items, workers, label, or_limit, oai_limit,
             cb_min=40, cb_rate=0.5, cb_consec=25):
    """Run items concurrently. Three safety halts:
       - BUDGET: OR/OAI spend over limit.
       - CIRCUIT BREAKER (rate): once >=cb_min non-skip attempts and cumulative error rate > cb_rate.
       - CIRCUIT BREAKER (consecutive): cb_consec failures in a row -> systemic failure (e.g. the 403
         key-limit flood that burned 40k attempts before this existed). Catches it in seconds even if
         many earlier calls succeeded."""
    done = skipped = errors = consec = 0; t0 = time.time(); last_err_type = ""
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *it): it for it in items}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                r = fut.result(); skipped += (r == "skip"); done += (r != "skip")
                if r != "skip": consec = 0  # a real success resets the consecutive counter
            except Exception as e:
                errors += 1; consec += 1; last_err_type = f"{type(e).__name__}: {e}"
                if errors <= 5: print(f"  [{label}] ERROR {futs[fut]}: {last_err_type}", flush=True)
            attempts = done + errors  # skips don't count toward the breaker
            if consec >= cb_consec or (attempts >= cb_min and errors / attempts > cb_rate):
                why = (f"{consec} consecutive failures" if consec >= cb_consec
                       else f"{errors}/{attempts} attempts failed (>{cb_rate:.0%})")
                print(f"  🛑 CIRCUIT BREAKER: {why}. Aborting. Last error: {last_err_type}", flush=True)
                ex.shutdown(wait=False, cancel_futures=True); return done, skipped, errors, True
            if i % 50 == 0 or i == len(futs):
                ok, o, a = budget_ok(or_limit, oai_limit)
                print(f"  [{label}] {i}/{len(futs)} done={done} skip={skipped} err={errors} "
                      f"| {time.time()-t0:.0f}s | OR ${o:.2f}/{or_limit:.0f} OAI ${a:.2f}/{oai_limit:.0f}", flush=True)
                if not ok:
                    print(f"  🛑 BUDGET HALT (OR ${o:.2f}>{or_limit} or OAI ${a:.2f}>{oai_limit}). Stopping.", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True); return done, skipped, errors, True
    return done, skipped, errors, False

def assemble_csv(out_dir, specs, personas, rc):
    state_cols = list(OUTCOMES.keys())
    for s in specs:
        if not (out_dir / s.name).exists():
            print(f"  [assemble] skip {s.name}: no output dir (model produced no rows)", flush=True)
            continue
        rows = []
        for p in personas:
            eid = str(p["employee_id"])
            for di in range(1, rc.n_days_per_persona + 1):
                fp = out_dir / s.name / f"{eid}_day{di}.json"
                if not fp.exists(): continue
                d = json.loads(fp.read_text())
                rows.append([eid, di, d.get("day_of_week", ""), d.get("day_type", ""),
                             *[d["states"].get(c, "") for c in state_cols],
                             d.get("metadata", {}).get("n_fallback", "")])
        with open(out_dir / s.name / "states.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["employee_id", "day_index", "day_of_week", "day_type",
                                           *state_cols, "n_fallback"]); w.writerows(rows)

def read_roster(path):
    specs = []
    for r in csv.DictReader(open(path)):
        name = r["model_id"].split("/")[-1].upper().replace("-", "_").replace(".", "_")
        specs.append(ModelSpec(name, r["model_id"], r["provider"]))
    return specs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roster", type=Path, default=REPO / "data/seeds/model_roster.csv")
    ap.add_argument("--event-model", default="gpt-4.1-mini")
    ap.add_argument("--event-provider", default="openai")
    ap.add_argument("--n-days", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--or-limit", type=float, default=50.0, help="OpenRouter halt threshold")
    ap.add_argument("--oai-limit", type=float, default=50.0, help="OpenAI halt threshold")
    ap.add_argument("--only-models", nargs="+", default=None, help="subset of model_ids to rate")
    a = ap.parse_args()

    rc = RunConfig(n_days_per_persona=a.n_days)
    # The seeder writes ONE ROW PER PERSONA-DAY (inverse-corpus schema). The harness
    # generates the N days internally, so we need ONE ROW PER PERSONA — dedup by id.
    # Build a per-(employee_id, day) indicator lookup from the FULL per-day rows so the
    # narrative is conditioned on each day's sampled indicators; dedup personas for identity.
    INDIC_PREFIX = ("behavioral", "physical", "social", "psychological", "digital", "life")
    indicator_lookup, seen, personas = {}, set(), []
    rdr = csv.DictReader(open(a.personas))
    icols = [c for c in (rdr.fieldnames or []) if c.split("_")[0] in INDIC_PREFIX]
    for r in rdr:
        e = r["employee_id"]
        try: d = int(float(r.get("day", 0)))
        except (ValueError, TypeError): d = 0
        indicator_lookup[(e, d)] = {c: r[c] for c in icols}
        if e in seen: continue
        seen.add(e); personas.append(r)
    if a.limit: personas = personas[:a.limit]
    ev_spec = ModelSpec("EVENT", a.event_model, a.event_provider)
    specs = read_roster(a.roster)
    if a.only_models: specs = [s for s in specs if s.model_id in a.only_models]
    ndir = a.out / "_shared_events"; ndir.mkdir(parents=True, exist_ok=True)

    for s in [ev_spec, *specs]:
        env = s.api_key_env()
        if env != "NONE" and not os.environ.get(env): sys.exit(f"Missing ${env} for {s.model_id}")

    print(f"== GEN: {len(personas)} personas x {a.n_days} days on {ev_spec.model_id} (workers={a.workers}) ==", flush=True)
    ev_client = make_client(ev_spec)
    _, _, _, halt = pool_run(lambda p: gen_persona(p, ev_client, ev_spec, rc, ndir, indicator_lookup),
                             [(p,) for p in personas], a.workers, "gen", a.or_limit, a.oai_limit)
    if halt: sys.exit("Halted in GEN on budget.")

    print(f"== RATE: {len(specs)} models x {len(personas)} personas x {a.n_days} days ==", flush=True)
    clients = {s.name: make_client(s) for s in specs}
    units = [(str(p["employee_id"]), di, s, clients[s.name], ndir, a.out)
             for s in specs for p in personas for di in range(1, a.n_days + 1)]
    print(f"   {len(units)} rating units", flush=True)
    _, _, _, halt = pool_run(rate_unit, units, a.workers, "rate", a.or_limit, a.oai_limit)

    assemble_csv(a.out, specs, personas, rc)
    ok, o, a2 = budget_ok(a.or_limit, a.oai_limit)
    print(f"== DONE. OpenRouter ${o:.2f} / OpenAI ${a2:.2f} | states.csv assembled per model ==", flush=True)

if __name__ == "__main__":
    main()
