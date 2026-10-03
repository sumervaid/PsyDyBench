#!/usr/bin/env python3
"""
extract_effect.py  

From the OpenAlex candidate set for each anchor target, identify the meta-analytic
effect and write a logged verification record. This is the step that turns the
unverified META_BENCHMARKS values into an auditable claim.

Design choices (no LLM dependency — deterministic + transparent)
----------------------------------------------------------------
1. Identify the best meta-analytic candidate:
   - prefer candidates whose title/abstract contain a meta-analysis cue
     ("meta-analy", "meta analysis", "systematic review");
   - among those, prefer one matching the asserted citation's first author + year
     (a "citation match"), else the highest-relevance meta-analytic hit.
2. Extract a verified_meta_r from that candidate's abstract:
   - regex for reported correlations / corrected rho (rho = .NN, r = .NN, "rho-hat"),
     and sample sizes (N = NNNN, k = NN studies). We do NOT fabricate a value: if no
     numeric effect is present in the abstract we leave verified_meta_r = null and set
     status accordingly (abstracts often omit the focal r — that is a needs-PDF flag,
     not a refutation).
3. Assign a status:
     verified  — meta-analytic source found AND |claimed - verified| <= TOL AND signs agree
     corrected — meta-analytic source found, an effect extracted, but it disagrees
                 with the claimed value beyond TOL (or sign flip)
     source_found_effect_in_pdf — right meta-analysis found, but the focal r is not in
                 the abstract (needs full-text human read; claim is plausibly sourced)
     unfound   — no credible meta-analytic source surfaced via OpenAlex

This keeps the human-in-the-loop honest: the script never upgrades a claim to
"verified" without an abstract-level numeric corroboration, and never silently
discards a claim whose source it located but whose number lives in the PDF.

Output (one file per pair):
    data/scraped/extractions/<indicator>__<state>.json

Usage:
    python3 scripts/scraping/extract_effect.py
    python3 scripts/scraping/extract_effect.py --pairs a:b,c:d   # trial subset
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data/scraped/raw"
EXT_DIR = REPO / "data/scraped/extractions"

TOL = 0.15  # |claimed - verified| within this -> "verified" (anchoring is direction/
            # magnitude-band oriented, not exact replication of a single rho)

META_CUES = ("meta-analy", "meta analysis", "meta-regression", "systematic review")

# Unicode hyphen/dash variants seen in publisher titles (e.g. "Meta‐Analytic" uses
# U+2010); normalize them to ASCII '-' so the meta-analysis cue matches.
_HYPHENS = "‐‑‒–—−"


def _norm(text):
    """Lowercase + normalize unicode hyphens/dashes to ASCII '-'."""
    if not text:
        return ""
    for h in _HYPHENS:
        text = text.replace(h, "-")
    return text.lower()


def parse_citation_hint(citation):
    """First-author surname + year from a META_BENCHMARKS citation string."""
    year = None
    m = re.search(r"\((\d{4})", citation)
    if m:
        year = m.group(1)
    lead = citation.strip()
    author = None
    if not lead.lower().startswith(
        ("conceptual", "inferred", "resource theory", "spillover", "circumplex")
    ):
        m2 = re.match(r"([A-Z][A-Za-z\-']+)", lead)
        if m2:
            author = m2.group(1)
    return author, year


def is_meta(cand):
    txt = _norm(f"{cand.get('title') or ''} {cand.get('abstract') or ''}")
    return any(cue in txt for cue in META_CUES)


def citation_match(cand, author, year):
    """Does this candidate plausibly equal the asserted citation?"""
    if not author:
        return False
    hay = f"{cand.get('title') or ''} {cand.get('authorships_first') or ''}".lower()
    a_ok = author.lower() in _norm(hay)
    # year-tolerant (+/-1) to absorb online-first vs print-citation drift.
    if year is None or cand.get("year") is None:
        y_ok = True
    else:
        try:
            y_ok = abs(int(cand["year"]) - int(year)) <= 1
        except (TypeError, ValueError):
            y_ok = str(cand.get("year")) == str(year)
    return a_ok and y_ok


# correlation-like effect sizes in an abstract, e.g. "r = .39", "ρ = -.21",
# "rho = 0.47", "corrected correlation of .42"
R_PATTERNS = [
    re.compile(r"(?:corrected\s+)?(?:correlation|rho|ρ|r)\s*(?:=|of|was)\s*(-?\.?\d?\.?\d{1,3})", re.I),
    re.compile(r"(-?0?\.\d{2})\s*\(?(?:corrected|true|population)\s*(?:correlation|rho|ρ|r)\)?", re.I),
]
N_PATTERN = re.compile(r"\bN\s*=\s*([\d,]{2,})", re.I)
K_PATTERN = re.compile(r"\bk\s*=\s*(\d{1,4})\b")
K_STUDIES = re.compile(r"(\d{1,4})\s+(?:studies|samples|independent samples)", re.I)


def extract_numeric_r(abstract):
    """Return the first plausible correlation magnitude in the abstract, else None.

    Conservative: we only return values in [-1, 1] and ignore obvious p-values.
    """
    if not abstract:
        return None
    for pat in R_PATTERNS:
        for m in pat.finditer(abstract):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            # ".39" parsed as 0.39; tolerate leading-dot forms
            if abs(val) <= 1.0 and abs(val) >= 0.01:
                return val
    return None


def extract_n(abstract):
    if not abstract:
        return None
    m = N_PATTERN.search(abstract)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def extract_k(abstract):
    if not abstract:
        return None
    for pat in (K_PATTERN, K_STUDIES):
        m = pat.search(abstract)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def best_candidate(record):
    """Pick the best meta-analytic candidate for this pair.

    Returns (candidate_or_None, citation_matched_bool).
    """
    author, year = parse_citation_hint(record["citation"])
    metas = [c for c in record["candidates"] if is_meta(c)]
    if not metas:
        return None, False
    # 1) citation match among metas
    for c in metas:
        if citation_match(c, author, year):
            return c, True
    # 2) fall back to highest relevance meta-analytic hit
    metas_sorted = sorted(
        metas, key=lambda c: (c.get("relevance_score") or 0), reverse=True
    )
    return metas_sorted[0], False


def evidence_quote(abstract, max_len=320):
    """A short snippet around the first effect-size mention for the audit log."""
    if not abstract:
        return ""
    for pat in R_PATTERNS:
        m = pat.search(abstract)
        if m:
            s = max(0, m.start() - 120)
            e = min(len(abstract), m.end() + 120)
            return abstract[s:e].strip()
    return abstract[:max_len].strip()


def classify(claimed, verified):
    """Status from claimed vs verified r."""
    if verified is None:
        return None  # caller handles 'effect in pdf'
    if (claimed < 0) != (verified < 0) and abs(claimed) > 0.05 and abs(verified) > 0.05:
        return "corrected"  # sign flip
    if abs(claimed - verified) <= TOL:
        return "verified"
    return "corrected"


def extract_pair(record):
    claimed = float(record["claimed_meta_r"])
    cand, cite_matched = best_candidate(record)

    if cand is None:
        return {
            "indicator": record["indicator"],
            "state": record["state"],
            "claimed_meta_r": claimed,
            "verified_meta_r": None,
            "source_doi": None,
            "source_title": None,
            "source_year": None,
            "citation_claimed": record["citation"],
            "citation_matched": False,
            "evidence_quote": "",
            "sample_n": None,
            "n_studies_k": None,
            "confidence_claimed": record["confidence"],
            "confidence_extraction": "none",
            "status": "unfound",
            "extracted_utc": datetime.now(timezone.utc).isoformat(),
            "note": "No meta-analytic candidate surfaced via OpenAlex for this pair.",
        }

    abstract = cand.get("abstract") or ""
    verified = extract_numeric_r(abstract)
    status = classify(claimed, verified)
    note = ""
    if status is None:
        status = "source_found_effect_in_pdf"
        note = (
            "Meta-analytic source located but no focal correlation present in the "
            "abstract; requires full-text human extraction."
        )

    # extraction confidence: high if citation matched AND a number was found;
    # medium if number found but citation not matched; low if source only.
    if verified is not None and cite_matched:
        ext_conf = "high"
    elif verified is not None:
        ext_conf = "medium"
    else:
        ext_conf = "low"

    return {
        "indicator": record["indicator"],
        "state": record["state"],
        "claimed_meta_r": claimed,
        "verified_meta_r": round(verified, 3) if verified is not None else None,
        "source_doi": cand.get("doi"),
        "source_title": cand.get("title"),
        "source_year": cand.get("year"),
        "source_openalex_id": cand.get("openalex_id"),
        "citation_claimed": record["citation"],
        "citation_matched": cite_matched,
        "evidence_quote": evidence_quote(abstract),
        "sample_n": extract_n(abstract),
        "n_studies_k": extract_k(abstract),
        "confidence_claimed": record["confidence"],
        "confidence_extraction": ext_conf,
        "abs_deviation": (
            round(abs(claimed - verified), 3) if verified is not None else None
        ),
        "status": status,
        "extracted_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None, help="comma-sep indicator:state subset")
    args = ap.parse_args()

    EXT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if args.pairs:
        wanted = set(
            p.strip().replace(":", "__") for p in args.pairs.split(",")
        )
        raw_files = [f for f in raw_files if f.stem in wanted]

    if not raw_files:
        print("No raw candidate files found. Run search_meta_analyses.py first.")
        return

    summary = []
    for f in raw_files:
        record = json.loads(f.read_text())
        ext = extract_pair(record)
        out = EXT_DIR / f.name
        out.write_text(json.dumps(ext, indent=2))
        summary.append(ext)
        v = ext["verified_meta_r"]
        v_str = f"{v:+.2f}" if v is not None else "  n/a"
        print(
            f"  {ext['status']:<28s} claimed {ext['claimed_meta_r']:+.2f} "
            f"verified {v_str}  {f.stem}"
        )

    # console tally
    from collections import Counter

    tally = Counter(s["status"] for s in summary)
    print("\nStatus tally:", dict(tally))


if __name__ == "__main__":
    main()
