#!/usr/bin/env python3
"""
verify_paper_values.py — recompute every number reported in the paper from the shipped
result files and check it against the LaTeX macro the manuscript actually uses.

Why this exists
---------------
The manuscript never hard-codes a result in prose: every reported quantity is a macro
defined in results/paper/values.tex (e.g. \nomoDir, \bestVarFid). This script is the
audit trail for those macros. For each one it (a) recomputes the value from the CSV/JSON
artifacts in this repository, (b) compares it to the macro at the precision the paper
prints, and (c) names the source file so a reader can follow any number back to its
origin. It exits non-zero if any recomputed value disagrees with the manuscript.

Three outcome labels are used:
  OK        recomputed value matches the macro at printed precision
  MISMATCH  recomputed value differs (this fails the script)
  UNUSED    the macro is defined in values.tex but the manuscript does not reference it

Usage:  python scripts/verify_paper_values.py [--write-doc]
        --write-doc also rewrites the table inside docs/PROVENANCE.md between the markers.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "validation"))
sys.path.insert(0, str(REPO / "scripts" / "magnitude"))
from meta_analysis_validation import META_BENCHMARKS, compute_within_r  # noqa: E402

LB  = REPO / "results/leaderboard"
MAG = REPO / "results/magnitude"
TEX = REPO / "results/paper/values.tex"

# ----------------------------------------------------------------------------- helpers

def read_macros(path: Path) -> dict:
    """Parse \\newcommand{\\name}{value} pairs; several may share one line."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", path.read_text()):
        out[m.group(1)] = m.group(2)
    return out

def tex_to_number(s: str):
    """'84.3\\%' -> 84.3 ; '1{,}800' -> 1800 ; 'nine' -> 9 ; non-numeric -> original string."""
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    t = s.replace("\\%", "").replace("{,}", "").strip()
    if t.lower() in words:
        return words[t.lower()]
    try:
        return float(t) if "." in t else int(t)
    except ValueError:
        return s

def printed_decimals(s: str) -> int:
    t = s.replace("\\%", "").replace("{,}", "")
    return len(t.split(".")[1]) if "." in t else 0

def pretty_model(tag: str) -> str:
    """GEMMA_2_27B_IT -> Gemma-2-27B (how the paper prints a model label)."""
    parts = [p for p in tag.split("_") if p != "IT"]
    return "-".join(p.capitalize() if p.isalpha() else p for p in parts)

DEVELOPER = {  # model-name prefix -> developer; used for the "nine developers" count
    "CLAUDE": "Anthropic", "DEEPSEEK": "DeepSeek", "GEMINI": "Google", "GEMMA": "Google",
    "GPT": "OpenAI", "GRANITE": "IBM", "LLAMA": "Meta", "MISTRAL": "Mistral", "PHI": "Microsoft",
    "QWEN": "Alibaba",
}
def developer(tag: str) -> str:
    return DEVELOPER[re.match(r"[A-Z]+", tag).group(0)]   # letters-only prefix: QWEN3_14B -> QWEN

def sign_match_pct(meta, obs) -> float:
    meta, obs = np.asarray(meta, float), np.asarray(obs, float)
    ok = ~np.isnan(obs)
    return 100.0 * np.mean(np.sign(meta[ok]) == np.sign(obs[ok]))

def icc1(df: pd.DataFrame, value: str, group_cols) -> float:
    """One-way random-effects ICC(1,1): agreement among repeated ratings of the same
    persona-day. Unbalanced groups handled with the standard k0 correction."""
    g = df.groupby(group_cols)[value]
    n_i = g.size().values; N = n_i.sum(); a = len(n_i)
    grand = df[value].mean()
    ssb = float((n_i * (g.mean().values - grand) ** 2).sum())
    ssw = float(((df[value] - g.transform("mean")) ** 2).sum())
    msb, msw = ssb / (a - 1), ssw / (N - a)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)

def within_r_from_ratings(ratings: pd.DataFrame, personas: pd.DataFrame) -> float:
    """Mean |within-person r| over the anchor pairs, for a ratings table with one row per
    persona-day (reps already averaged). Mirrors scripts/magnitude/eval_within_r.py."""
    r = ratings.rename(columns={"day_index": "day"})
    r = r.rename(columns={s: f"{s}_state" for s in
                          ["work_engagement","psych_safety","work_creativity","team_trust"]})
    p = personas.rename(columns={"day_index": "day"})
    df = r.merge(p, on=["employee_id", "day"], how="inner")
    vals = []
    for b in META_BENCHMARKS:
        if b["indicator"] in df.columns and b["state"] in df.columns:
            v = compute_within_r(df, b["indicator"], b["state"])
            if v is not None and not np.isnan(v):
                vals.append(abs(v))
    return float(np.mean(vals))

# ----------------------------------------------------------------------------- compute

def recompute() -> dict:
    """Return {macro: (value, source)} for every macro we can derive from shipped files."""
    R = {}
    lb    = pd.read_csv(LB / "leaderboard.csv")
    lb2   = pd.read_csv(LB / "leaderboard_v2.csv")
    nomo  = pd.read_csv(LB / "nomological_leaderboard.csv")
    anch  = pd.read_csv(LB / "anchoring_leaderboard.csv")
    cult  = pd.read_csv(LB / "cultural_validation.csv")
    # The paper's analysed roster is every model except two corpora excluded as degenerate at the
    # final run: MISTRAL_SMALL_24B_INSTRUCT_2501 produced a one-row corpus (no within-person
    # statistics) and QWEN3_5_9B is a variance-error outlier (3.33, an order of magnitude above the
    # rest). The convergence means in the paper were taken over the full anchoring file.
    DEGENERATE = {"MISTRAL_SMALL_24B_INSTRUCT_2501", "QWEN3_5_9B"}
    clean = lb[~lb.model.isin(DEGENERATE)].copy()          # the 34 analysed corpora
    anch_c = anch

    src_lb, src_nomo, src_anch = "results/leaderboard/leaderboard.csv", \
        "results/leaderboard/nomological_leaderboard.csv", "results/leaderboard/anchoring_leaderboard.csv"

    R["nModels"]    = (len(lb), src_lb)
    R["nAnalyzed"]  = (len(clean), src_lb + " minus the two degenerate corpora")
    full = lb2[lb2.n_personas == lb2.n_personas.max()].iloc[0]
    R["nPersonas"]  = (int(full.n_personas), "results/leaderboard/leaderboard_v2.csv")
    R["nDays"]      = (int(full.n_obs // full.n_personas), "results/leaderboard/leaderboard_v2.csv")
    R["nStates"]    = (4, "scripts/generation/gen_config.py (OUTCOMES)")
    R["nDevelopers"]= (len({developer(m) for m in clean.model}), src_nomo + " (model prefixes)")
    R["paramLo"]    = (f"{int(nomo.params_b.min())}B", src_nomo)
    R["paramHi"]    = (f"{int(nomo.params_b.max())}B", src_nomo)
    R["nomoDir"]    = (nomo.nomo_dir_pct.mean(), src_nomo)
    R["nomoRangeLo"]= (nomo.nomo_dir_pct.min(), src_nomo)
    R["nomoRangeHi"]= (nomo.nomo_dir_pct.max(), src_nomo)
    R["betweenConv"]= (anch_c.between_dir_pct.mean(), src_anch + " (mean over all rows)")
    R["withinConv"] = (anch_c.within_dir_pct.mean(), src_anch + " (mean over all rows)")
    best = clean.loc[clean.var_fidelity.idxmin()]
    R["bestVarFid"]      = (best.var_fidelity, src_lb)
    R["bestVarFidModel"] = (pretty_model(best.model), src_lb)
    R["gptFiveVarFid"]   = (float(lb.loc[lb.model == "GPT_5_5", "var_fidelity"].iloc[0]), src_lb)
    R["gemmaFourVarFid"] = (float(lb.loc[lb.model == "GEMMA_3_4B_IT", "var_fidelity"].iloc[0]), src_lb)
    R["gemmaFourConv"]   = (float(anch.loc[anch.model == "GEMMA_3_4B_IT", "pooled_dir_pct"].iloc[0]), src_anch)
    R["nLevels"]         = (6, "scripts/validation/internal_validity_v2.py (L1,L2,L3,L4,L6,L7)")
    for lvl, mac in [("L2","lTwoPass"), ("L7","lSevenPass"), ("L4","lFourPass")]:
        R[mac] = (int((clean[lvl] == "P").sum()), src_lb)
    R["maxLevelsPassed"] = (int(clean.levels_passed.max()), src_lb)
    R["varErrMedian"]    = (float(clean.var_fidelity.median()), src_lb)
    R["nAnchors"]        = (len(META_BENCHMARKS), "scripts/validation/meta_analysis_validation.py")
    R["cultMinInvar"]    = (cult.invariance_mean_abs_dr.min(), "results/leaderboard/cultural_validation.csv")
    R["cultMaxInvar"]    = (cult.invariance_mean_abs_dr.max(), "results/leaderboard/cultural_validation.csv")
    R["cultConv"]        = (cult.xcult_conv_dirmatch_pct.max(), "results/leaderboard/cultural_validation.csv")

    # --- magnitude probe (appendix): reliability, baseline within-person magnitude, de-anchoring
    ratings = pd.read_csv(MAG / "ratings_GPT_4_1.csv")
    t8 = ratings[ratings.temperature == 0.8]
    rel = [icc1(t8, s, ["employee_id","day_index"]) for s in
           ["work_engagement","psych_safety","work_creativity","team_trust"]]
    R["relYMean"] = (float(np.mean(rel)), "results/magnitude/ratings_GPT_4_1.csv (ICC(1,1) over re-ratings, T=0.8)")
    base = pd.read_csv(MAG / "pairs_V0_baseline_full.csv")
    R["withinRbaseline"] = (float(base.within_r.abs().mean()), "results/magnitude/pairs_V0_baseline_full.csv")
    R["dirFull"]  = (sign_match_pct(base.meta_r, base.within_r), "results/magnitude/pairs_V0_baseline_full.csv")
    R["rhoFull"]  = (float(stats.spearmanr(base.meta_r, base.within_r, nan_policy="omit").correlation),
                     "results/magnitude/pairs_V0_baseline_full.csv")
    ss = pd.read_csv(MAG / "search_summary.csv").set_index("variant")
    R["nObsFull"]    = (int(ss.loc["V5_combo_full","n_obs"]), "results/magnitude/search_summary.csv")
    R["dirDeanchor"] = (float(ss.loc["V5_combo_full","dirmatch_within_pct"]), "results/magnitude/search_summary.csv")
    R["rhoDeanchor"] = (float(ss.loc["V5_combo_full","rho_within_vs_meta"]), "results/magnitude/search_summary.csv")
    personas = pd.read_csv(REPO / "data/seeds/personas_full.csv")
    import tempfile, eval_within_r as EWR  # the paper's own evaluator (scripts/magnitude/eval_within_r.py)
    for T, mac in [(0.8,"withinRtempEight"), (0.0,"withinRtempZero")]:
        sub = ratings[ratings.temperature == T].groupby(["employee_id","day_index"], as_index=False)[
            ["work_engagement","psych_safety","work_creativity","team_trust"]].mean()
        tmp = Path(tempfile.gettempdir()) / f"ratings_T{T}.csv"; sub.to_csv(tmp, index=False)
        summ, _ = EWR.evaluate(str(tmp))
        val = summ["mean_abs_within_r"] if isinstance(summ, dict) else getattr(summ, "mean_abs_within_r")
        R[mac] = (float(val), f"results/magnitude/ratings_GPT_4_1.csv (T={T}) via scripts/magnitude/eval_within_r.py")
    for tag, rho_mac, dir_mac in [("GPT_4_1_MINI","miniRho",None), ("GPT_4_1_NANO","nanoRho","nanoDir")]:
        d = pd.read_csv(MAG / f"pairs_V0_baseline_{tag}.csv"); srcf = f"results/magnitude/pairs_V0_baseline_{tag}.csv"
        R[rho_mac] = (float(stats.spearmanr(d.meta_r, d.within_r, nan_policy="omit").correlation), srcf)
        if dir_mac: R[dir_mac] = (sign_match_pct(d.meta_r, d.within_r), srcf)
    # --- low-ICC construct-variance experiment
    ns = pd.read_csv(MAG / "newstates_within_r.csv")
    import new_states as NS  # noqa: E402  (literature ICC values for the three added states)
    icc_of = {k: v.get("icc") for k, v in NS.NEW_STATES.items()}
    for key, stem in [("daily_exhaustion_state","exhaust"), ("daily_fatigue_state","fatigue"), ("daily_flow_state","flow")]:
        d = ns[ns.state == key]; srcf = "results/magnitude/newstates_within_r.csv"
        R[f"{stem}ICC"]     = (icc_of.get(key), "scripts/magnitude/new_states.py (literature ICC)")
        R[f"{stem}WithinR"] = (float(d.within_r.abs().mean()), srcf)
        if stem != "flow": R[f"{stem}Dir"] = (sign_match_pct(d.meta_r, d.within_r), srcf)
        if stem == "fatigue": R["fatigueRho"] = (float(stats.spearmanr(d.meta_r, d.within_r).correlation), srcf)
    R["nIccPairs"] = (int(ns.groupby("state").size().iloc[0]), "results/magnitude/newstates_within_r.csv")
    # --- MIPROv2 achievability tune
    for tag, stem in [("mini","mini"), ("nano","nano")]:
        j = json.load(open(MAG / f"dspy_tune_gpt_4_1_{tag}.json")); srcf = f"results/magnitude/dspy_tune_gpt_4_1_{tag}.json"
        R[f"{stem}BaseComp"]  = (j["baseline"], srcf); R[f"{stem}TunedComp"] = (j["tuned"], srcf)
        R["nTuneTrain"] = (j["n_train"], srcf); R["nTuneTest"] = (j["n_test"], srcf)
    cs = json.load(open(REPO / "data/seeds/cross_state_targets.json"))["pairs"]
    pairs = cs.values() if isinstance(cs, dict) else cs
    # L3 cross-state targets: pairs whose meta-analytic source was located (verified, full-text
    # verified, or abstract-only), as opposed to pairs left as expert priors.
    R["nVerifiedL"] = (sum(1 for p in pairs if p.get("status") not in (None, "no_meta_found", "expert_prior", "placeholder")), "data/seeds/cross_state_targets.json (pairs with a located source)")
    return R

# Macros defined in values.tex that the camera-ready manuscript does not reference. They are
# reported as UNUSED rather than verified or failed.
UNUSED = {"nStates", "lFourPass", "varErrMedian", "nAnchors", "anchorsVerified"}

# ----------------------------------------------------------------------------- compare

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write-doc", action="store_true"); a = ap.parse_args()
    macros = read_macros(TEX); R = recompute()
    rows, bad = [], 0
    for name, raw in macros.items():
        tex = tex_to_number(raw)
        if name in UNUSED and name not in R:
            rows.append((name, raw, "—", "", "UNUSED")); continue
        if name in R:
            val, src = R[name]
            if isinstance(tex, str) or isinstance(val, str):
                ok = str(val).strip() == str(tex).strip()
                shown = str(val)
            else:
                dec = printed_decimals(raw); shown = f"{val:.{max(dec,3)}f}"
                ok = round(float(val), dec) == round(float(tex), dec)
            status = ("OK" if ok else "MISMATCH") + (" (unused in manuscript)" if name in UNUSED else "")
            bad += (not ok) and name not in UNUSED
        else:
            shown, src, status = "—", "", "UNTRACED"; bad += 1
        rows.append((name, raw, shown, src, status))
    table = ["| macro | value in paper | recomputed | source | status |", "|---|---|---|---|---|"]
    table += [f"| `\\{n}` | `{t}` | {s} | `{src}` | {st} |" for n, t, s, src, st in rows]
    print("\n".join(table)); print(f"\n{len(rows)} macros; {bad} mismatched/untraced")
    if a.write_doc:
        doc = REPO / "docs/PROVENANCE.md"; txt = doc.read_text()
        start, end = "<!-- BEGIN AUTO-TABLE -->", "<!-- END AUTO-TABLE -->"
        txt = txt[:txt.index(start)+len(start)] + "\n" + "\n".join(table) + "\n" + txt[txt.index(end):]
        doc.write_text(txt)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
