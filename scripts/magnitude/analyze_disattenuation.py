#!/usr/bin/env python3
"""
Magnitude experiment — Stage 2: reliability + disattenuation.

Reads the re-rating CSV from run_openai_reliability.py and computes, per state:
  - single-rating reliability rel_Y = ICC(1,1) across repeated ratings of the SAME narrative
    (one-way random-effects: rel = (MSB - MSW) / (MSB + (k-1)*MSW))
  - Spearman-Brown projection rel_M for M averaged ratings
Then disattenuates the EXISTING pooled within-person |r| from the recovery baselines
(results/phase14/recovery/per_pair_<RATER>.csv) via r_true = r_obs / sqrt(rel_Y),
since indicators are IID-seeded (rel_X ~ 1). For gpt-4.1 also reports reliability by temperature.

Outputs (no overwrite of prior results; new files):
  results/magnitude/reliability_by_state_<RATER>.csv
  results/magnitude/disattenuation_<RATER>.csv
  results/magnitude/MAGNITUDE_RESULTS.md  (appends a per-rater section)
"""
import argparse, csv, math, statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATES = ["work_engagement", "psych_safety", "work_creativity", "team_trust"]
# map rater -> existing per-pair recovery file (within_r baseline)
PERPAIR = {"gpt-4.1": "per_pair_GPT41.csv", "gpt-5.5": "per_pair_GPT51.csv"}


def icc1(groups):
    """ICC(1,1) one-way random effects across groups (each group = repeated ratings of one narrative)."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    k = st.mean(len(g) for g in groups)
    grand = st.mean(x for g in groups for x in g)
    n = len(groups)
    msb = sum(len(g) * (st.mean(g) - grand) ** 2 for g in groups) / (n - 1)
    msw = sum((x - st.mean(g)) ** 2 for g in groups for x in g) / sum(len(g) - 1 for g in groups)
    denom = msb + (k - 1) * msw
    return (msb - msw) / denom if denom > 0 else None


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def reliability(rows, temp_filter=None):
    """rel_Y per state across reps of each (emp,day). temp_filter: e.g. '0.8' or 'default'."""
    out = {}
    for s in STATES:
        groups = defaultdict(list)
        for r in rows:
            if temp_filter is not None and str(r["temperature"]) != str(temp_filter):
                continue
            v = r.get(s, "")
            if v not in ("", None):
                groups[(r["employee_id"], r["day_index"])].append(float(v))
        out[s] = icc1(list(groups.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gpt-4.1", "gpt-5.5"])
    args = ap.parse_args()
    name = args.model.upper().replace("-", "_").replace(".", "_")
    rows = load_rows(REPO / f"results/magnitude/ratings_{name}.csv")
    outdir = REPO / "results/magnitude"; outdir.mkdir(parents=True, exist_ok=True)

    # production-temp reliability (gpt-4.1 -> 0.8; gpt-5.5 -> default)
    prod = "0.8" if args.model == "gpt-4.1" else "default"
    rel = reliability(rows, temp_filter=prod)

    with open(outdir / f"reliability_by_state_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["state", "rel_Y_single", "rel_3", "rel_5", "rel_10"])
        for s in STATES:
            r1 = rel[s]
            sb = lambda m, r: (m * r / (1 + (m - 1) * r)) if r else None
            w.writerow([s, _r(r1), _r(sb(3, r1)), _r(sb(5, r1)), _r(sb(10, r1))])

    # disattenuation against existing within_r baselines
    pp = REPO / "results/phase14/recovery" / PERPAIR[args.model]
    dis_rows = []
    if pp.exists():
        with open(pp) as f:
            for r in csv.DictReader(f):
                s = r["state"].replace("_state", "")
                ry = rel.get(s)
                wr = _f(r.get("within_r"))
                if ry and wr is not None and ry > 0:
                    dis_rows.append((r["indicator"], s, wr, ry, wr / math.sqrt(ry), _f(r.get("meta_r"))))
    with open(outdir / f"disattenuation_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["indicator", "state", "within_r_obs", "rel_Y", "within_r_corrected", "meta_r"])
        for row in dis_rows:
            w.writerow([row[0], row[1], _r(row[2]), _r(row[3]), _r(row[4]), _r(row[5])])

    obs_abs = st.mean(abs(r[2]) for r in dis_rows) if dis_rows else None
    cor_abs = st.mean(abs(r[4]) for r in dis_rows) if dis_rows else None
    rel_mean = st.mean(v for v in rel.values() if v) if any(rel.values()) else None

    # temperature sweep (gpt-4.1 only)
    temp_lines = []
    if args.model == "gpt-4.1":
        for t in ["0.0", "0.3", "0.8"]:
            rt = reliability(rows, temp_filter=t)
            m = st.mean(v for v in rt.values() if v) if any(rt.values()) else None
            temp_lines.append((t, m))

    # append a results section
    md = outdir / "MAGNITUDE_RESULTS.md"
    with open(md, "a") as f:
        f.write(f"\n## {args.model} — reliability & disattenuation\n\n")
        f.write(f"- n rating rows: {len(rows)}; production-temp reliability filter: `{prod}`\n")
        f.write(f"- single-rating reliability rel_Y per state: "
                + ", ".join(f"{s}={_r(rel[s])}" for s in STATES) + f" (mean {_r(rel_mean)})\n")
        f.write(f"- pooled within-person |r|: observed {_r(obs_abs)} -> disattenuated {_r(cor_abs)} "
                f"(human meta band 0.20-0.70)\n")
        if temp_lines:
            f.write("- reliability by rating temperature (gpt-4.1): "
                    + ", ".join(f"t={t}:{_r(m)}" for t, m in temp_lines) + "\n")
    print(f"rel_Y mean={_r(rel_mean)} | within|r| obs={_r(obs_abs)} -> corrected={_r(cor_abs)}")
    if temp_lines:
        print("temp sweep reliability:", temp_lines)


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def _r(x):
    return round(x, 3) if isinstance(x, (int, float)) else ""


if __name__ == "__main__":
    main()
