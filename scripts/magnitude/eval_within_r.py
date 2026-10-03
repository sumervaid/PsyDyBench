#!/usr/bin/env python3
"""
Evaluate within-person magnitude recovery for a ratings CSV — methodologically identical
to the paper (reuses META_BENCHMARKS + compute_within_r from validation/meta_analysis_validation).

Input ratings CSV columns: employee_id, day_index (or day), work_engagement, psych_safety,
work_creativity, team_trust  (one row per persona-day; if multiple rows/reps, they are averaged).
Joins the seeded daily indicators (data/seeds/personas_full.csv) and computes, over the anchored
indicator-state pairs: mean |within_r|, within-person direction-match %, and per-pair table.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "validation"))

# meta_analysis_validation imports plotting libs at module load that we don't need here.
# Stub them so the (pure pandas/numpy/scipy) correlation functions import cleanly anywhere.
import types  # noqa: E402
_mpl = types.ModuleType("matplotlib"); _mpl.__path__ = []
_plt = types.ModuleType("matplotlib.pyplot")
_fm = types.ModuleType("matplotlib.font_manager")
class _FM:  # fm.fontManager.addfont(...) no-op
    def addfont(self, *a, **k): pass
_fm.fontManager = _FM()
_lines = types.ModuleType("matplotlib.lines"); _lines.Line2D = object
_mpl.pyplot = _plt; _mpl.font_manager = _fm; _mpl.lines = _lines; _mpl.use = lambda *a, **k: None
for _n, _m in [("matplotlib", _mpl), ("matplotlib.pyplot", _plt),
               ("matplotlib.font_manager", _fm), ("matplotlib.lines", _lines)]:
    sys.modules.setdefault(_n, _m)
_at = types.ModuleType("adjustText"); _at.adjust_text = lambda *a, **k: None
sys.modules.setdefault("adjustText", _at)
sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))

from meta_analysis_validation import META_BENCHMARKS, compute_within_r, compute_pooled_r  # noqa: E402

STATE_MAP = {"work_engagement": "work_engagement_state", "psych_safety": "psych_safety_state",
             "work_creativity": "work_creativity_state", "team_trust": "team_trust_state"}


def evaluate(ratings_csv, personas_csv=None):
    personas_csv = personas_csv or (REPO / "data/seeds/personas_full.csv")
    s = pd.read_csv(ratings_csv)
    if "day" not in s.columns and "day_index" in s.columns:
        s = s.rename(columns={"day_index": "day"})
    keep = [c for c in STATE_MAP if c in s.columns]
    s = s.groupby(["employee_id", "day"])[keep].mean().reset_index()      # average any reps
    s = s.rename(columns={k: STATE_MAP[k] for k in keep})
    ind = pd.read_csv(personas_csv)
    # personas_full.csv carries placeholder *_state columns; drop them so the rated
    # states merge in cleanly (otherwise pandas suffixes them and anchors never match).
    ind = ind.drop(columns=[c for c in STATE_MAP.values() if c in ind.columns], errors="ignore")
    merged = ind.merge(s, on=["employee_id", "day"], how="inner")
    rows = []
    for e in META_BENCHMARKS:
        i, st, mr = e["indicator"], e["state"], e["meta_r"]
        if i not in merged.columns or st not in merged.columns:
            continue
        rows.append({"indicator": i, "state": st, "meta_r": mr,
                     "pooled_r": compute_pooled_r(merged, i, st),
                     "within_r": compute_within_r(merged, i, st),
                     "confidence": e.get("confidence", "")})
    df = pd.DataFrame(rows)
    vw = df.dropna(subset=["within_r"])
    hi = vw[vw["confidence"] == "high"] if "confidence" in vw.columns else vw
    summary = {
        "n_obs": len(merged), "n_employees": merged["employee_id"].nunique(),
        "n_pairs": len(vw),
        "mean_abs_within_r": float(vw["within_r"].abs().mean()) if len(vw) else float("nan"),
        "mean_abs_within_r_highconf": float(hi["within_r"].abs().mean()) if len(hi) else float("nan"),
        "dirmatch_within_pct": float(100 * (np.sign(vw["within_r"]) == np.sign(vw["meta_r"])).mean()) if len(vw) else float("nan"),
        "rho_within_vs_meta": float(vw["within_r"].corr(vw["meta_r"], method="spearman")) if len(vw) > 2 else float("nan"),
    }
    return summary, df


if __name__ == "__main__":
    import json
    summ, df = evaluate(sys.argv[1])
    print(json.dumps(summ, indent=2))
