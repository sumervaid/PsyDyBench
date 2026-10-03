#!/usr/bin/env python3
"""
Build the MANUAL-DOWNLOAD worklist: the exhaustive, DOI-deduplicated list of source papers
whose focal effect size could NOT be auto-verified (paywalled, OA-but-unreadable, or read but
the pairwise r was not in running text), so a human can pull the PDF and extract the r.

Groups by unique paper (source_doi) so one download can resolve multiple anchor pairs.
Reads:  data/scraped/extractions/*.json
Writes: results/anchors/MANUAL_DOWNLOAD_WORKLIST.{md,csv}
Usage:  python scripts/scraping/build_manual_worklist.py
"""
import json, glob, csv, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data/scraped/extractions"
OUT = REPO / "results/anchors"; OUT.mkdir(parents=True, exist_ok=True)

# statuses that still need a human (verified/corrected are DONE; unfound has no source to fetch)
NEEDS = {
    "source_found_effect_in_pdf": "paywalled (no OA copy)",
    "oa_pdf_unreadable": "OA URL unreadable (landing page / 403 / scanned)",
    "oa_effect_not_found": "OA read but pairwise r not in extracted text",
    "abstract_unverified_suspect": "abstract-step value is a suspected artifact",
}

def doi_url(doi):
    if not doi: return ""
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"

def main():
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(EXT / "*.json")))]
    need = [r for r in recs if r["status"] in NEEDS and r.get("source_doi")]
    nosrc = [r for r in recs if r["status"] in ("unfound",) or not r.get("source_doi")]

    # group by DOI -> unique paper
    bydoi = collections.OrderedDict()
    for r in need:
        bydoi.setdefault(r["source_doi"], []).append(r)

    # csv: one row per (paper, pair)
    with open(OUT / "MANUAL_DOWNLOAD_WORKLIST.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doi_url", "source_title", "source_year", "indicator", "state",
                    "claimed_meta_r", "fail_reason", "status"])
        for doi, rs in bydoi.items():
            for r in rs:
                w.writerow([doi_url(doi), r.get("source_title", ""), r.get("source_year", ""),
                            r["indicator"], r["state"], r.get("claimed_meta_r"),
                            NEEDS[r["status"]], r["status"]])

    lines = ["# Manual-download worklist — anchor papers needing human effect-size extraction", "",
             f"Generated from `data/scraped/extractions/`. **{len(bydoi)} unique papers** cover "
             f"**{len(need)} anchor pairs** that automated OA retrieval could not verify.", "",
             "For each paper: open the DOI, find the correlation table, read off the focal r for the "
             "listed (indicator → state) pair(s), and record it. Papers are sorted by how many pairs "
             "they unlock (download these first).", ""]
    ranked = sorted(bydoi.items(), key=lambda kv: -len(kv[1]))
    for i, (doi, rs) in enumerate(ranked, 1):
        t = rs[0].get("source_title", "?"); y = rs[0].get("source_year", "")
        lines.append(f"### {i}. {t} ({y}) — {len(rs)} pair(s)")
        lines.append(f"- DOI: {doi_url(doi)}")
        lines.append(f"- Access: {NEEDS[rs[0]['status']]}")
        for r in rs:
            lines.append(f"  - `{r['indicator']}` → `{r['state']}`  (claimed r = {r.get('claimed_meta_r')})")
        lines.append("")
    if nosrc:
        lines.append(f"## No source located ({len(nosrc)} pairs) — need a fresh literature search, not a download")
        for r in nosrc:
            lines.append(f"- `{r['indicator']}` → `{r['state']}` (claimed r = {r.get('claimed_meta_r')}; "
                         f"claimed cite: {r.get('citation_claimed','?')})")
    (OUT / "MANUAL_DOWNLOAD_WORKLIST.md").write_text("\n".join(lines) + "\n")

    print(f"unique papers needing manual download: {len(bydoi)}")
    print(f"anchor pairs they cover: {len(need)}")
    print(f"pairs with NO source located (need search): {len(nosrc)}")
    print("by reason:")
    for k, v in collections.Counter(NEEDS[r['status']] for r in need).most_common():
        print(f"  {v:3d}  {k}")
    print(f"-> {OUT/'MANUAL_DOWNLOAD_WORKLIST.md'}")

if __name__ == "__main__":
    main()
