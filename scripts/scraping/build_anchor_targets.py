#!/usr/bin/env python3
"""
build_anchor_targets.py  

Read the META_BENCHMARKS table that backs the meta-analytic convergence anchoring
framework and emit a flat, machine-readable list of (indicator, state) anchor
targets to be verified against the primary literature via OpenAlex.

The META_BENCHMARKS table currently carries a banner warning that its values are
"from web search — verify against originals before publication". This script is the
first link in the reproducible verification chain:

    build_anchor_targets.py  -> data/seeds/anchor_targets.csv
    search_meta_analyses.py  -> data/scraped/raw/<indicator>__<state>.json
    extract_effect.py        -> data/scraped/extractions/<indicator>__<state>.json

Output columns (anchor_targets.csv):
    indicator         the synthetic context-engine indicator column name
    state             one of the 4 states (WE / PS / WC / TT)
    claimed_meta_r    the effect size asserted in META_BENCHMARKS (to be checked)
    citation          the source asserted in META_BENCHMARKS (to be checked)
    construct_match   how the indicator was mapped to the meta-analytic construct
    confidence        the asserted confidence in the construct match (high/medium/low)

No network access. Pure read of the in-repo table -> CSV.

Usage:
    python3 scripts/scraping/build_anchor_targets.py
"""

import csv
import importlib.util
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
META_SCRIPT = REPO / "scripts/validation/meta_analysis_validation.py"
OUT_CSV = REPO / "data/seeds/anchor_targets.csv"


def load_meta_benchmarks():
    """Import META_BENCHMARKS from the validation script without running its main().

    meta_analysis_validation.py imports matplotlib/adjustText/scipy at module load,
    which we do not need here. We read only the literal list. To stay robust to those
    heavy imports we load the module via importlib but guard against import failure by
    falling back to a literal-AST extraction.
    """
    try:
        spec = importlib.util.spec_from_file_location("meta_validation", META_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # noqa: executes module-level code (defines the list)
        return mod.META_BENCHMARKS
    except Exception as exc:  # heavy/optional deps missing -> parse the literal instead
        print(f"[info] direct import failed ({exc}); falling back to AST literal parse")
        return _parse_benchmarks_via_ast()


def _parse_benchmarks_via_ast():
    """Extract the META_BENCHMARKS list literal from source without importing deps."""
    import ast

    src = META_SCRIPT.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "META_BENCHMARKS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("META_BENCHMARKS not found in meta_analysis_validation.py")


def main():
    benchmarks = load_meta_benchmarks()
    print(f"Loaded {len(benchmarks)} (indicator, state) pairs from META_BENCHMARKS")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "indicator",
        "state",
        "claimed_meta_r",
        "citation",
        "construct_match",
        "confidence",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for entry in benchmarks:
            w.writerow(
                {
                    "indicator": entry["indicator"],
                    "state": entry["state"],
                    "claimed_meta_r": entry["meta_r"],
                    "citation": entry["citation"],
                    "construct_match": entry["construct_match"],
                    "confidence": entry["confidence"],
                }
            )

    # quick confidence breakdown for the console
    from collections import Counter

    conf = Counter(e["confidence"] for e in benchmarks)
    print(f"Wrote {len(benchmarks)} rows -> {OUT_CSV}")
    print(f"  confidence: {dict(conf)}")


if __name__ == "__main__":
    main()
