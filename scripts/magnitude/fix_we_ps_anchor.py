#!/usr/bin/env python3
"""Resolve the WE×PS L3 placeholder: replace the 0.50 expert prior with the full-text-verified
Frazier (2017) Engagement-row value (rho=0.45) already recorded in the file. Data correction; logged
in ANALYTIC_DECISIONS_LEDGER.md. Run once."""
import json
from pathlib import Path

f = Path(__file__).resolve().parents[2] / "data/seeds/cross_state_targets.json"
d = json.loads(f.read_text())
p = d["pairs"]["work_engagement__psych_safety"]
before = {"r": p["r"], "is_expert_prior": p["is_expert_prior"], "margin": p["margin"], "status": p.get("status")}
p["r"] = 0.45
p["is_expert_prior"] = False
p["status"] = "full_text_verified"
p["margin"] = 0.10
p["confidence"] = "medium"
p["resolution_note"] = ("2026-06-22: replaced 0.50 expert prior with the full-text-verified Frazier 2017 "
                        "Table 2 Engagement-row corrected rho=0.45 (already recorded as rho_corrected_fulltext); "
                        "is_expert_prior set False, margin tightened to 0.10.")
f.write_text(json.dumps(d, indent=4))
print("WE×PS BEFORE:", before)
print("WE×PS AFTER :", {"r": p["r"], "is_expert_prior": p["is_expert_prior"], "margin": p["margin"], "status": p["status"]})
