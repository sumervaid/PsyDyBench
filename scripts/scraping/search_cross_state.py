#!/usr/bin/env python3
"""
search_cross_state.py  —  cross-state nomological-net sourcing helper

Locate, for each of the 6 pairwise BETWEEN-PERSON correlations among the four
workplace psychological states (work_engagement WE, psych_safety PS,
work_creativity WC, team_trust TT), a published meta-analysis (with DOI) that
reports the focal r. Reuses the repo's Stream-A pattern:
  - OpenAlex works API (free, no key) to find candidate meta-analyses
  - Unpaywall to find an OA full-text PDF where one exists
  - pdftotext (poppler) to extract text so the focal r can be human-confirmed

This script ONLY gathers candidates + OA PDF text snippets to disk for human
inspection; it does NOT write the final targets JSON (that is curated by hand
from the located evidence to honor the no-invented-number rule).

Output: results/anchors/cross_state_candidates/<pairkey>.json
        results/anchors/cross_state_candidates/<pairkey>__<doi>.txt  (OA fulltext, if any)

Usage: python3 scripts/scraping/search_cross_state.py
"""
import json
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/anchors/cross_state_candidates"
OPENALEX = "https://api.openalex.org/works"
MAILTO = "svaid@hbs.edu"
TIMEOUT = 30

# Search queries per pair: a few hand-built free-text queries aimed at the
# canonical OB meta-analyses that relate each state pair between-person.
PAIRS = {
    "work_engagement__psych_safety": [
        "work engagement psychological safety meta-analysis",
        "psychological safety engagement meta-analytic review",
        "Frazier psychological safety meta-analysis engagement",
    ],
    "work_engagement__work_creativity": [
        "work engagement creativity meta-analysis",
        "employee engagement innovation creativity meta-analytic",
    ],
    "work_engagement__team_trust": [
        "work engagement trust meta-analysis",
        "employee engagement organizational trust meta-analytic",
    ],
    "psych_safety__work_creativity": [
        "psychological safety creativity meta-analysis",
        "psychological safety innovation creativity meta-analytic review",
        "Frazier psychological safety meta-analysis creativity",
    ],
    "psych_safety__team_trust": [
        "psychological safety trust meta-analysis",
        "psychological safety team trust meta-analytic antecedents",
        "Frazier psychological safety meta-analysis trust antecedent",
    ],
    "work_creativity__team_trust": [
        "trust creativity meta-analysis workplace",
        "interpersonal trust creativity innovation meta-analytic",
        "team trust creativity performance meta-analysis",
    ],
}


def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos)) if pos else ""


def openalex_search(query, per_page=6):
    params = {
        "search": query,
        "filter": "type:article",
        "per-page": per_page,
        "sort": "relevance_score:desc",
        "mailto": MAILTO,
    }
    r = requests.get(OPENALEX, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        auths = w.get("authorships", [])
        first = (auths[0].get("author") or {}).get("display_name") if auths else None
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        out.append({
            "openalex_id": w.get("id"),
            "title": w.get("title"),
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
            "doi": w.get("doi"),
            "year": w.get("publication_year"),
            "oa_status": (w.get("open_access") or {}).get("oa_status"),
            "cited_by_count": w.get("cited_by_count"),
            "authorships_first": first,
            "source": src.get("display_name"),
            "relevance_score": w.get("relevance_score"),
            "matched_query": query,
        })
    return out


def unpaywall_pdf(doi):
    doi_clean = doi.replace("https://doi.org/", "")
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi_clean)}?email={MAILTO}"
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url), timeout=20))
    except Exception:
        return None
    if not d.get("is_oa"):
        return None
    loc = d.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")


def fetch_pdf_text(pdf_url):
    try:
        req = urllib.request.Request(
            pdf_url, headers={"User-Agent": "Mozilla/5.0 (research; svaid@hbs.edu)"})
        raw = urllib.request.urlopen(req, timeout=45).read()
        if len(raw) < 1000:
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tf:
            tf.write(raw)
            tf.flush()
            res = subprocess.run(
                ["pdftotext", tf.name, "-"], capture_output=True, timeout=60)
            txt = res.stdout.decode("utf-8", "ignore")
            return txt if len(txt) > 500 else None
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pairkey, queries in PAIRS.items():
        print(f"=== {pairkey} ===", flush=True)
        cands = []
        seen = set()
        for q in queries:
            try:
                for c in openalex_search(q):
                    cid = c["openalex_id"]
                    if cid not in seen:
                        seen.add(cid)
                        cands.append(c)
            except Exception as e:
                print(f"  [warn] {q!r}: {e}")
            time.sleep(0.4)
        cands.sort(key=lambda c: -(c.get("cited_by_count") or 0))
        rec = {
            "pair": pairkey,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "n_candidates": len(cands),
            "candidates": cands,
        }
        (OUT / f"{pairkey}.json").write_text(json.dumps(rec, indent=2))
        print(f"  {len(cands)} candidates")
        # try OA fulltext for the top 3 most-cited candidates with a DOI
        for c in cands[:3]:
            if not c.get("doi"):
                continue
            pdf = unpaywall_pdf(c["doi"])
            if not pdf:
                continue
            txt = fetch_pdf_text(pdf)
            if txt:
                safe = c["doi"].replace("https://doi.org/", "").replace("/", "_")
                (OUT / f"{pairkey}__{safe}.txt").write_text(txt)
                print(f"  OA fulltext saved: {c['doi']}")
            time.sleep(0.4)
    print("Done.")


if __name__ == "__main__":
    main()
