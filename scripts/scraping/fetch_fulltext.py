#!/usr/bin/env python3
"""
the pipeline, step 4 — automated OPEN-ACCESS full-text fetch + effect-size extraction.

For each anchor pair whose source was located but whose effect is "in the PDF" (not the
abstract), use Unpaywall (free, legal) to find an OA full-text PDF, download it, and
LLM-extract the focal meta-analytic r for that construct pair. Updates the extraction JSON
in place. Paywalled papers (no OA copy) are left untouched (status unchanged).

  DOI --Unpaywall--> OA PDF URL --download--> pymupdf text --gpt-4.1-mini--> {r, quote}

Usage:  python scripts/scraping/fetch_fulltext.py            # all DOI-bearing pairs
        python scripts/scraping/fetch_fulltext.py --workers 6
"""
import os, sys, json, glob, argparse, threading, urllib.request, urllib.parse
import concurrent.futures as cf
from pathlib import Path
import fitz  # pymupdf
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
# self-contained .env load (OPENAI_API_KEY) so this runs standalone on any host
_envf = REPO / ".env"
if _envf.exists():
    for _l in _envf.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
EXT = REPO / "data/scraped/extractions"
BUD = REPO / "results/budget"; BUD.mkdir(parents=True, exist_ok=True)
EMAIL = "svaid@hbs.edu"
_LOCK = threading.Lock()
_client = OpenAI()  # OpenAI direct (gpt-4.1-mini); cheap

def log_usage(tin, tout):
    with _LOCK:
        with open(BUD / "openai_usage.jsonl", "a") as f:
            f.write(json.dumps({"model": "gpt-4.1-mini", "prompt_tokens": int(tin or 0),
                                "completion_tokens": int(tout or 0)}) + "\n")

def unpaywall_pdf(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url), timeout=20))
    except Exception:
        return None
    if not d.get("is_oa"):
        return None
    loc = d.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")

def fetch_text(pdf_url):
    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (research; svaid@hbs.edu)"})
        raw = urllib.request.urlopen(req, timeout=40).read()
        if len(raw) < 1000 or len(raw) > 40_000_000:
            return None
        doc = fitz.open(stream=raw, filetype="pdf")
        txt = "\n".join(p.get_text() for p in doc)
        return txt if len(txt) > 500 else None
    except Exception:
        return None

def readable(col):
    """behavioral_break_minutes -> 'break minutes'; work_creativity_state -> 'work creativity'."""
    s = col
    for p in ("behavioral_", "physical_", "social_", "psychological_", "digital_", "life_work_", "life_"):
        if s.startswith(p): s = s[len(p):]; break
    if s.endswith("_state"): s = s[:-6]
    return s.replace("_", " ")

def construct_descriptor(d):
    return (f'the construct "{readable(d["indicator"])}" and the psychological state '
            f'"{readable(d["state"])}" (paper: {d.get("source_title","")[:120]})')

def llm_extract(text, construct_match, claimed_r):
    chunk = text[:60000]
    sys_p = ("You read meta-analyses and extract the focal meta-analytic correlation (r) "
             "for a specified construct relationship. Be literal; do not guess.")
    user_p = (f"This is the full text of a meta-analysis. Find the META-ANALYTIC correlation "
              f"(typically a sample-size-weighted mean r, often in a table) for the relationship: "
              f"{construct_match}. Return STRICT JSON: "
              f'{{"found": true/false, "r": <number or null>, "quote": "<the sentence or table '
              f'cell, <=200 chars>"}}. If the specific pairwise meta-analytic r is not reported, '
              f'found=false. (The claimed value to check is {claimed_r}.)\n\nTEXT:\n{chunk}')
    r = _client.chat.completions.create(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
    log_usage(getattr(r.usage, "prompt_tokens", 0), getattr(r.usage, "completion_tokens", 0))
    t = r.choices[0].message.content or ""
    try:
        s = t[t.index("{"):t.rindex("}") + 1]
        return json.loads(s)
    except Exception:
        return {"found": False, "r": None, "quote": ""}

def process(fp):
    d = json.loads(Path(fp).read_text())
    if not d.get("source_doi") or d.get("status") == "verified":
        return "skip"
    pdf_url = unpaywall_pdf(d["source_doi"])
    if not pdf_url:
        d["oa"] = False; Path(fp).write_text(json.dumps(d, indent=2)); return "no_oa"
    d["oa"] = True; d["oa_pdf_url"] = pdf_url
    text = fetch_text(pdf_url)
    if not text:
        d["status"] = "oa_pdf_unreadable"; Path(fp).write_text(json.dumps(d, indent=2)); return "unreadable"
    res = llm_extract(text, construct_descriptor(d), d.get("claimed_meta_r"))
    if res.get("found") and res.get("r") is not None:
        vr = float(res["r"]); claimed = d.get("claimed_meta_r")
        d["verified_meta_r"] = vr; d["evidence_quote"] = res.get("quote", "")[:200]
        d["confidence_extraction"] = "full_text"
        agree = claimed is not None and (vr * claimed > 0) and abs(abs(vr) - abs(claimed)) < 0.15
        d["status"] = "verified" if agree else "corrected"
        Path(fp).write_text(json.dumps(d, indent=2)); return d["status"]
    d["status"] = "oa_effect_not_found"; Path(fp).write_text(json.dumps(d, indent=2)); return "oa_no_effect"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    files = [f for f in sorted(glob.glob(str(EXT / "*.json")))
             if json.loads(Path(f).read_text()).get("source_doi")]
    print(f"{len(files)} DOI-bearing pairs to attempt", flush=True)
    import collections; c = collections.Counter(); done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, f): f for f in files}
        for fut in cf.as_completed(futs):
            try: c[fut.result()] += 1
            except Exception as e: c["error"] += 1
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  {done}/{len(files)} {dict(c)}", flush=True)
    print("DONE", dict(c), flush=True)

if __name__ == "__main__":
    main()
