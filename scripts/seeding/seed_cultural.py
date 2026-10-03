#!/usr/bin/env python3
"""
Culture-conditioned forward seeding (all-silico). For each nation in cultural_norms.json,
apply its Hofstede-informed Big Five offsets (Schmitt et al. 2007 nation-level z-shifts) to the
published trait MEANS, seed N personas with the existing forward seeder, and concatenate.
Culture is encoded in employee_id (e.g. USA_SYN-0001) so downstream validation recovers it with
no schema change. States remain NaN (forward; filled by generation).

Usage: python3 seed_cultural.py --n 40 --days 6 --out ../../data/seeds/personas_cultural.csv
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
import seed_personas as sp  # noqa

REPO = SD.parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="personas PER culture")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "data/seeds/personas_cultural.csv")
    a = ap.parse_args()

    norms = sp.load_norms(sp.DEFAULT_NORMS)
    cultures = json.load(open(REPO / "data/seeds/cultural_norms.json"))["cultures"]
    sds = {t: norms["big_five"]["sds"][t] for t in sp.BIG5_ORDER}

    frames = []
    for ci, (cname, cv) in enumerate(cultures.items()):
        cn = copy.deepcopy(norms)
        for t in sp.BIG5_ORDER:  # shift trait means by offset (in SD units)
            cn["big_five"]["means"][t] = norms["big_five"]["means"][t] + cv["big5_offset"].get(t, 0.0) * sds[t]
        rng = np.random.default_rng(a.seed + ci)
        df, _, _ = sp.build_dataframe(rng, cn, a.n, a.days)
        df["employee_id"] = cname + "_" + df["employee_id"].astype(str)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)[sp.SCHEMA_COLUMNS]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}  shape={out.shape}  cultures={list(cultures)}  "
          f"personas={out.employee_id.nunique()}  days={a.days}")

if __name__ == "__main__":
    main()
