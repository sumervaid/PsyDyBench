#!/usr/bin/env python3
"""
Validation leaderboard: run the internal-validity battery on every model's generated
corpus and assemble the leaderboard. Bridges the the pipeline norms.json -> the pipeline internal_validity.py
schema (adds derived between/within SDs + literature cross-state correlations; renames the
corpus columns work_engagement->work_engagement_state, day_index->day).

Usage: python scripts/validation/run_validation.py results/generation/full
Outputs: results/leaderboard/internal_validity_targets.json, <MODEL>_internal_validity.json, leaderboard.csv
"""
import csv, json, glob, math, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NORMS = json.load(open(REPO / "data/seeds/norms.json"))
OUT = REPO / "results/leaderboard"; OUT.mkdir(parents=True, exist_ok=True)
BATTERY = REPO / "scripts/validation/internal_validity.py"

STATE_MAP = {"work_engagement": "work_engagement_state", "psych_safety": "psych_safety_state",
             "work_creativity": "work_creativity_state", "team_trust": "team_trust_state"}
ORDER = list(STATE_MAP.values())

# Literature-informed cross-state correlation targets (DOCUMENTED, to be finalised via
# the pipeline anchor verification). Between-person: moderate positive among engagement/safety/
# creativity/trust (~0.40); within-person: ~0.25 (the internal-validity battery spec empirical centre). Off-diagonal
# uniform pending per-pair lit values.
BETWEEN_R, WITHIN_R = 0.40, 0.25

def build_targets():
    states = {}
    for s, st in NORMS["states"].items():
        icc = st["icc"]; tsd = st["total_sd"]
        states[s] = {  # internal_validity reads targets["states"] by BARE state name (STATE_NAMES)
            "grand_mean": st["grand_mean"], "grand_sd": tsd, "icc": icc,
            "between_person_sd": round(tsd * math.sqrt(icc), 4),
            "within_person_sd": round(tsd * math.sqrt(1 - icc), 4),
        }
    def mat(off):
        return [[1.0 if i == j else off for j in range(4)] for i in range(4)]
    t = {"states": states, "state_col_order": ORDER,
         "between_person_state_corr": mat(BETWEEN_R), "within_person_state_corr": mat(WITHIN_R),
         "_provenance": "grand_mean/total_sd/icc from data/seeds/norms.json (cited); between/within "
                        "SD derived; cross-state corr literature-informed placeholders (between 0.40 / "
                        "within 0.25) pending the pipeline finalisation."}
    (OUT / "internal_validity_targets.json").write_text(json.dumps(t, indent=2))
    return OUT / "internal_validity_targets.json"

def adapt_corpus(states_csv, dest):
    rows = list(csv.DictReader(open(states_csv)))
    cols = ["employee_id", "day"] + ORDER
    with open(dest, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r["employee_id"], r["day_index"]] + [r.get(k, "") for k in STATE_MAP])
    return dest

def target_within_sd():
    # keyed by BARE state name (matches the raw states.csv columns)
    return {s: NORMS["states"][s]["total_sd"] * math.sqrt(1 - NORMS["states"][s]["icc"])
            for s in NORMS["states"]}

def variance_fidelity(states_csv):
    """Graded discriminator: mean |ln(within_sd / target_within_sd)| across states.
    0 = perfect; higher = worse (over- or under-varying). This is what the battery's binary
    L2 collapses, and it's where the models actually separate."""
    import collections, statistics as st
    rows = list(csv.DictReader(open(states_csv)))
    twsd = target_within_sd()
    devs = []
    for s, col in twsd.items():
        byp = collections.defaultdict(list)
        for r in rows:
            v = r.get(s)
            if v: byp[r["employee_id"]].append(float(v))
        sds = [st.pstdev(v) for v in byp.values() if len(v) > 1]
        if sds:
            wsd = st.mean(sds)
            devs.append(abs(math.log(wsd / col)) if wsd > 0 and col > 0 else float("nan"))
    return round(st.mean([d for d in devs if d == d]), 4) if devs else None

def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/generation/full")
    targets = build_targets()
    board = []
    for sc in sorted(glob.glob(str(base / "*/states.csv"))):
        model = Path(sc).parent.name
        adapted = adapt_corpus(sc, OUT / f"_corpus_{model}.csv")
        outj = OUT / f"{model}_internal_validity.json"
        subprocess.run([sys.executable, str(BATTERY), "--corpus", str(adapted),
                        "--targets", str(targets), "--out", str(outj)], check=True,
                       capture_output=True)
        d = json.load(open(outj))
        row = {"model": model, "internal_validity": d["overall_score"],
               "levels_passed": d["levels_passed"],
               "var_fidelity": variance_fidelity(sc)}  # lower = closer to literature variance
        for lv in d["levels"]:
            row[f"L{lv['level']}"] = "P" if lv["passed"] else "."
        board.append(row)
    board.sort(key=lambda r: (r["var_fidelity"] if r["var_fidelity"] is not None else 9))
    cols = ["model", "var_fidelity", "internal_validity", "levels_passed", "L1", "L2", "L3", "L4", "L6", "L7"]
    with open(OUT / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in board: w.writerow({k: r.get(k, "") for k in cols})
    print(f"{'model':32s} {'varFid↓':>7s} {'battery':>7s} {'lvls':>5s}  L1 L2 L3 L4 L6 L7")
    for r in board:
        vf = f"{r['var_fidelity']:.3f}" if r['var_fidelity'] is not None else "  n/a"
        print(f"{r['model']:32s} {vf:>7s} {r['internal_validity']:>7.3f} {r['levels_passed']:>3d}/6   "
              + "  ".join(r.get(f'L{l}', '?') for l in [1, 2, 3, 4, 6, 7]))
    print("\nvarFid = mean |ln(within-SD / literature target)| across 4 states; 0=perfect, lower=better.")

if __name__ == "__main__":
    main()
