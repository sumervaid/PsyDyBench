#!/usr/bin/env python3
"""Generate the appendix model-leaderboard LaTeX table (both criteria) from the result CSVs.
Internal validity: within-person variance error (lower=better) + internal-validity criteria passed.
External validity: trait->state direction agreement + rank rho. Ranked by internal (var error)."""
import csv
from pathlib import Path
A = Path(__file__).resolve().parents[2]
PAPER = A / "results" / "paper"
PAPER.mkdir(parents=True, exist_ok=True)
def f(x):
    try: return float(x)
    except: return None
lb = {r["model"]: r for r in csv.DictReader(open(A/"results/leaderboard/leaderboard.csv"))}
nm = {r["model"]: r for r in csv.DictReader(open(A/"results/leaderboard/nomological_leaderboard.csv"))}

def clean(m):
    s = m.replace("_INSTRUCT","").replace("_IT","").replace("_2507","").replace("_"," ").title()
    s = s.replace("Gpt","GPT").replace(" ","-")
    # restore decimals: Gemma-2-27B etc fine; Qwen2-5 -> Qwen2.5 (best-effort)
    return s
rows=[]
for m,r in lb.items():
    ve=f(r.get("var_fidelity"))
    if ve is None or ve>1.0: continue   # drop degenerate corpora
    n=nm.get(m,{})
    rows.append({"name":clean(m),"params":n.get("params_b","") or "--","ve":ve,
                 "lv":r.get("levels_passed",""),"dir":f(n.get("nomo_dir_pct")),"rho":f(n.get("nomo_rho"))})
rows.sort(key=lambda x:x["ve"])
lines=[]
lines.append(r"\begin{table}[t]")
lines.append(r"\centering\footnotesize")
lines.append(r"\caption{Model leaderboard across both validity criteria, ranked by internal validity "
             r"(within-person variance error, lower is closer to the human norm). External validity "
             r"(trait--state direction agreement and rank correlation $\rho$) stays high and roughly flat "
             r"as internal validity varies, the central split of the benchmark.}")
lines.append(r"\label{app:leaderboard}")
lines.append(r"\begin{tabular}{l r r c r r}")
lines.append(r"\toprule")
lines.append(r"Model & Params (B) & Var.\ error $\downarrow$ & Levels & Dir.\ (\%) & $\rho$ \\")
lines.append(r"\midrule")
for x in rows:
    d = f"{x['dir']:.0f}" if x['dir'] is not None else "--"
    rho = f"{x['rho']:.2f}" if x['rho'] is not None else "--"
    lines.append(f"{x['name']} & {x['params']} & {x['ve']:.3f} & {x['lv']}/6 & {d} & {rho} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(PAPER/"model_ranking_table.tex").write_text("\n".join(lines)+"\n")
print(f"wrote model_ranking_table.tex with {len(rows)} models (ranked by var error)")
print("top3:", [(x['name'],round(x['ve'],3),x['lv']) for x in rows[:3]])
print("bottom3:", [(x['name'],round(x['ve'],3),x['lv']) for x in rows[-3:]])
