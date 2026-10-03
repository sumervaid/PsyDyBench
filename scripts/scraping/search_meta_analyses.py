#!/usr/bin/env python3
"""
search_meta_analyses.py  

For each (indicator, state) anchor target, query the OpenAlex works API for the
meta-analysis that backs the claimed effect size, and save the raw candidate set
(title, abstract, doi, year, oa_status, ...) to disk for the extraction step.

OpenAlex is free, requires no API key, and we identify ourselves via the polite
pool (mailto=). See https://docs.openalex.org/.

Query construction
------------------
We build a search string from the construct_match field (which names the two
constructs the meta-analytic r relates) plus an explicit "meta-analysis" cue, and
restrict to journal articles. We also parse the asserted citation (e.g.
"Crawford, LePine & Rich (2010) JAP") into a first-author + year hint and run a
SECOND, citation-targeted query so the extraction step can prefer the exact paper
the META_BENCHMARKS row claims.

Output (one file per pair):
    data/scraped/raw/<indicator>__<state>.json
    {
      "indicator", "state", "claimed_meta_r", "citation", "construct_match",
      "confidence", "queries": [...], "retrieved_utc",
      "candidates": [ {title, abstract, doi, year, oa_status, cited_by_count,
                       authorships_first, openalex_id, type, source,
                       matched_query} , ... ]
    }

Usage:
    # all pairs in anchor_targets.csv
    python3 scripts/scraping/search_meta_analyses.py
    # restrict to a comma-sep list of indicator:state pairs (the 3-pair trial)
    python3 scripts/scraping/search_meta_analyses.py \
        --pairs psychological_autonomy_felt:work_engagement_state,\
social_conflict_instances:psych_safety_state,\
social_manager_interaction_quality:team_trust_state
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
TARGETS_CSV = REPO / "data/seeds/anchor_targets.csv"
RAW_DIR = REPO / "data/scraped/raw"

OPENALEX = "https://api.openalex.org/works"
MAILTO = "svaid@hbs.edu"
PER_PAGE = 8           # candidates per query
SLEEP_S = 0.4          # be polite; OpenAlex polite pool is generous
TIMEOUT = 30

# Map state column -> readable construct phrase used in the search string.
STATE_PHRASE = {
    "work_engagement_state": "work engagement",
    "psych_safety_state": "psychological safety",
    "work_creativity_state": "workplace creativity",
    "team_trust_state": "trust in team",
}


def reconstruct_abstract(inv_index):
    """OpenAlex returns abstracts as an inverted index {word: [positions]}.

    Rebuild the linear abstract text. Returns "" if absent.
    """
    if not inv_index:
        return ""
    positions = {}
    for word, idxs in inv_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def parse_citation(citation):
    """Pull a first-author surname + year hint from a META_BENCHMARKS citation string.

    e.g. "Crawford, LePine & Rich (2010) JAP" -> ("Crawford", "2010")
         "Conceptual: absorption subscale of UWES" -> (None, None)
    """
    year = None
    m = re.search(r"\((\d{4})", citation)
    if m:
        year = m.group(1)
    author = None
    # take the leading token before a comma / & / ( that looks like a surname
    lead = citation.strip()
    if lead.lower().startswith(("conceptual", "inferred", "resource theory", "spillover")):
        author = None
    else:
        m2 = re.match(r"([A-Z][A-Za-z\-']+)", lead)
        if m2:
            author = m2.group(1)
    return author, year


def build_construct_query(construct_match, state):
    """Search string from the construct mapping + a meta-analysis cue."""
    # construct_match looks like "Autonomy -> Engagement"; strip the arrow/parens
    cm = re.sub(r"[→\-]+>", " ", construct_match)
    cm = re.sub(r"[()]", " ", cm)
    cm = cm.replace("→", " ").replace("->", " ").replace("/", " ")
    cm = re.sub(r"\s+", " ", cm).strip()
    phrase = STATE_PHRASE.get(state, "")
    return f"{cm} {phrase} meta-analysis".strip()


STOPWORDS = {
    "the", "a", "of", "and", "to", "in", "for", "its", "with", "test", "an",
    "on", "at", "as", "review", "study", "meta", "analysis", "meta-analysis",
}


def title_search_terms(construct_match, state):
    """Build OpenAlex title.search terms for the claimed meta-analysis.

    The free-text 'search' endpoint buries canonical meta-analyses; a title.search
    filter (verified by hand to surface Christian 2011 & Dirks & Ferrin 2002 exactly,
    e.g. title.search:"trust in leadership meta-analytic") is far more precise.

    We seed title terms from the ANTECEDENT construct (left of the arrow in
    construct_match, which names the meta-analysis topic, e.g. "Transformational
    leadership") plus the state construct and a 'meta-analytic' cue. We do not store
    the paper title (only author+year), so this is a best-effort precision query that
    complements the construct + citation free-text queries.
    """
    # antecedent = text left of the arrow / 'inverse' qualifier
    ante = re.split(r"[→\-]+>|→|->", construct_match)[0]
    ante = re.sub(r"[()]", " ", ante)
    ante = re.sub(r"\s+", " ", ante).strip()
    words = [w for w in ante.split() if w.lower() not in STOPWORDS]
    state_word = {
        "work_engagement_state": "engagement",
        "psych_safety_state": "psychological safety",
        "work_creativity_state": "creativity",
        "team_trust_state": "trust",
    }.get(state, "")
    terms = " ".join(words + [state_word, "meta-analytic"]).strip()
    return terms


def openalex_search(query, per_page=PER_PAGE, extra_filter=None):
    """One OpenAlex works query (free-text); return normalized candidate dicts.

    extra_filter: optional additional OpenAlex filter clause(s), comma-joined onto
    the base 'type:article' (e.g. 'publication_year:2011').
    """
    filt = "type:article"
    if extra_filter:
        filt = f"{filt},{extra_filter}"
    params = {
        "search": query,
        "filter": filt,
        "per-page": per_page,
        "sort": "relevance_score:desc",
        "mailto": MAILTO,
    }
    r = requests.get(OPENALEX, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = []
    for w in data.get("results", []):
        auths = w.get("authorships", [])
        first = None
        if auths:
            first = (auths[0].get("author") or {}).get("display_name")
        loc = (w.get("primary_location") or {})
        src = (loc.get("source") or {})
        out.append(
            {
                "openalex_id": w.get("id"),
                "title": w.get("title"),
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                "doi": w.get("doi"),
                "year": w.get("publication_year"),
                "type": w.get("type"),
                "oa_status": (w.get("open_access") or {}).get("oa_status"),
                "cited_by_count": w.get("cited_by_count"),
                "authorships_first": first,
                "source": src.get("display_name"),
                "relevance_score": w.get("relevance_score"),
                "matched_query": query,
            }
        )
    return out


def openalex_title_search(title_terms, year=None, per_page=PER_PAGE):
    """Precision query via the title.search + publication_year filters.

    This is the lookup that reliably surfaces canonical meta-analyses (verified by
    hand for Christian et al. 2011 and Dirks & Ferrin 2002). Returns the same
    normalized candidate dicts as openalex_search.
    """
    if not title_terms:
        return []
    clauses = [f"title.search:{title_terms}"]
    if year:
        # OpenAlex publication_year often reflects online-first, which can differ by
        # +/-1 from the print citation year (e.g. Frazier et al. is indexed 2016 but
        # cited as 2017). Use a +/-1 range to avoid false 'unfound'.
        try:
            y = int(year)
            clauses.append(f"publication_year:{y-1}|{y}|{y+1}")
        except (TypeError, ValueError):
            clauses.append(f"publication_year:{year}")
    params = {
        "filter": ",".join(clauses),
        "per-page": per_page,
        "mailto": MAILTO,
    }
    r = requests.get(OPENALEX, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = []
    for w in data.get("results", []):
        auths = w.get("authorships", [])
        first = (auths[0].get("author") or {}).get("display_name") if auths else None
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        out.append(
            {
                "openalex_id": w.get("id"),
                "title": w.get("title"),
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                "doi": w.get("doi"),
                "year": w.get("publication_year"),
                "type": w.get("type"),
                "oa_status": (w.get("open_access") or {}).get("oa_status"),
                "cited_by_count": w.get("cited_by_count"),
                "authorships_first": first,
                "source": src.get("display_name"),
                "relevance_score": w.get("relevance_score"),
                "matched_query": f"title.search:{title_terms} (year={year})",
            }
        )
    return out


def dedupe(cands):
    """Drop duplicate OpenAlex IDs, keeping the first (highest-relevance) hit."""
    seen, out = set(), []
    for c in cands:
        cid = c.get("openalex_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


def search_pair(row):
    """Run construct + citation-targeted queries for one anchor target."""
    construct_q = build_construct_query(row["construct_match"], row["state"])
    queries = [construct_q]

    author, year = parse_citation(row["citation"])
    candidates = []

    # (a) construct free-text query
    try:
        candidates.extend(openalex_search(construct_q))
    except Exception as exc:
        print(f"    [warn] query failed: {construct_q!r} -> {exc}")
    time.sleep(SLEEP_S)

    # (b) citation free-text query (author + year + construct)
    if author and year:
        phrase = STATE_PHRASE.get(row["state"], "")
        cite_q = f"{author} {year} {phrase} meta-analysis"
        queries.append(cite_q)
        try:
            candidates.extend(openalex_search(cite_q, extra_filter=f"publication_year:{year}"))
        except Exception as exc:
            print(f"    [warn] query failed: {cite_q!r} -> {exc}")
        time.sleep(SLEEP_S)

    # (c) precision title.search queries restricted to the claimed year, with
    #     progressive term relaxation (title.search is AND across terms, so the full
    #     phrase often returns 0 when the paper's title omits a qualifier word).
    title_terms = title_search_terms(row["construct_match"], row["state"])
    if title_terms:
        toks = title_terms.split()
        variants = [title_terms]
        if len(toks) > 2:
            variants.append(" ".join(toks[1:]))          # drop leading qualifier
        if "meta-analytic" in toks:
            variants.append(" ".join(t for t in toks if t != "meta-analytic"))
        # broadest: just the state construct + 'meta-analytic' (the source meta may be
        # about the state generally, not the specific antecedent — e.g. Frazier 2016).
        state_only = {
            "work_engagement_state": "engagement meta-analytic",
            "psych_safety_state": "psychological safety meta-analytic",
            "work_creativity_state": "creativity meta-analytic",
            "team_trust_state": "trust meta-analytic",
        }.get(row["state"])
        if state_only:
            variants.append(state_only)
        seen_v = set()
        for v in variants:
            if v in seen_v or not v.strip():
                continue
            seen_v.add(v)
            tq = f"title.search:{v} year:{year}"
            queries.append(tq)
            hits = []
            try:
                hits = openalex_title_search(v, year=year)
                candidates.extend(hits)
            except Exception as exc:
                print(f"    [warn] title query failed: {tq!r} -> {exc}")
            time.sleep(SLEEP_S)
            if hits:
                break  # stop at the first relaxation level that returns anything

    candidates = dedupe(candidates)
    return {
        "indicator": row["indicator"],
        "state": row["state"],
        "claimed_meta_r": row["claimed_meta_r"],
        "citation": row["citation"],
        "construct_match": row["construct_match"],
        "confidence": row["confidence"],
        "queries": queries,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }


def load_targets():
    with TARGETS_CSV.open() as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs",
        help="comma-sep indicator:state pairs to restrict to (default: all)",
        default=None,
    )
    args = ap.parse_args()

    rows = load_targets()
    if args.pairs:
        wanted = set(p.strip() for p in args.pairs.split(","))
        rows = [r for r in rows if f"{r['indicator']}:{r['state']}" in wanted]
        print(f"Restricted to {len(rows)} pairs: {sorted(wanted)}")
    else:
        print(f"Processing all {len(rows)} pairs")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, 1):
        pair = f"{row['indicator']}__{row['state']}"
        print(f"[{i}/{len(rows)}] {pair}")
        result = search_pair(row)
        out = RAW_DIR / f"{pair}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"    {result['n_candidates']} candidates -> {out.name}")

    print("Done.")


if __name__ == "__main__":
    main()
