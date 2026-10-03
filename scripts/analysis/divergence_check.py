#!/usr/bin/env python3
"""
Quick integrity + cross-model divergence check on generated states.csv files.
Integrity: each model should have N personas x days unique rows, states in-range, no dupes.
Divergence: per-model mean + avg within-person SD per state (the fluctuation signal the
frameworks score). Wide spread across models => the frameworks can discriminate.

Usage: python scripts/analysis/divergence_check.py results/generation/full
"""
import csv, glob, sys, collections, statistics as st
from pathlib import Path

OUT = {"work_engagement": (1, 7), "psych_safety": (1, 4),
       "work_creativity": (1, 6), "team_trust": (1, 5)}
TARGET_WSD = {"work_engagement": 0.70, "psych_safety": 0.40,
              "work_creativity": 0.64, "team_trust": 0.45}

def within_sd(rows, c):
    byp = collections.defaultdict(list)
    for r in rows:
        if r.get(c): byp[r["employee_id"]].append(float(r[c]))
    sds = [st.pstdev(v) for v in byp.values() if len(v) > 1]
    return round(st.mean(sds), 3) if sds else None

def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/generation/full")
    print(f"targets within-SD: {TARGET_WSD}\n")
    hdr = f"{'model':32s} {'n':>5s} {'uniq_p':>6s} {'oob':>4s}"
    for c in OUT: hdr += f" {c[:4]+'_wsd':>9s}"
    print(hdr)
    for f in sorted(glob.glob(str(base / "*/states.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows: continue
        m = Path(f).parent.name
        uniq = len({(r["employee_id"], r["day_index"]) for r in rows})
        oob = sum(1 for r in rows for c, (lo, hi) in OUT.items()
                  if r.get(c) and not (lo <= float(r[c]) <= hi))
        line = f"{m:32s} {len(rows):>5d} {uniq:>6d} {oob:>4d}"
        for c in OUT: line += f" {str(within_sd(rows, c)):>9s}"
        print(line)
    print("\n(oob = out-of-bounds ratings; should be 0. uniq_p = unique persona-days.)")

if __name__ == "__main__":
    main()
