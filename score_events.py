"""STAGE 1 (slow, LLM, run once -> cached): point-in-time event scoring.

Pipeline:
  1. Resolve each S&P 100 ticker -> CIK via SEC's public map.
  2. Pull 8-K / 10-Q filings in the window from EDGAR (free, timestamped,
     reproducible — the cleanest point-in-time text source).
  3. For each filing, fetch the primary document text.
  4. Score it with the LLM on the TEXT ALONE (temperature=0, prompt forbids
     using any knowledge of later price action). This reduces *elicitation* of
     the model's hindsight; it does NOT eliminate parametric contamination —
     that is what the FinBERT baseline in stage 2 quantifies.
  5. Save the raw text per filing so FinBERT can score the IDENTICAL text.

Outputs (in ./cache/):
  events_panel.csv   columns: date,ticker,event_type,score,accession,summary
  texts/{accession}.txt   raw filing text (for the FinBERT baseline)
  scored.json        cache of already-scored accessions (resumable)

Requires network + an OpenAI-compatible key. Reads DEEPSEEK_API_KEY (or
OPENAI_API_KEY) and OPENAI_BASE_URL from the environment / agent/.env.

SEC compliance: a descriptive User-Agent with contact info is REQUIRED. Set
SEC_UA, e.g.  export SEC_UA="Your Name your@email.com"
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

import env_config  # noqa: F401  (loads .env into os.environ)

CACHE = Path(__file__).resolve().parent / "cache"
TEXTS = CACHE / "texts"
SEC_UA = os.getenv("SEC_UA", "eventalpha-research contact@example.com")
HEADERS = {"User-Agent": SEC_UA}
FORMS = {"8-K", "10-Q"}

# 8-K item codes worth scoring (informative). Everything else (5.03 bylaws,
# 7.01 Reg FD, 9.01 exhibits-only, etc.) is routine noise and is skipped.
# This is the signal-isolation step: don't dilute earnings/material events
# with boilerplate disclosures.
INFORMATIVE_ITEMS = {
    "1.01",  # entry into material agreement
    "1.03",  # bankruptcy / receivership
    "2.01",  # completion of acquisition / disposition
    "2.02",  # results of operations (earnings)
    "2.03",  # material direct financial obligation
    "2.04",  # triggering events accelerating obligations
    "4.01",  # auditor change
    "4.02",  # non-reliance / restatement (strong negative)
    "5.02",  # director / officer departures & appointments
}
# 10-Q is always kept (it's a substantive quarterly report).

# S&P 100 (OEX) — edit freely. A static list is fine for a backtest; note any
# survivorship caveat in your writeup.
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN","AVGO","AXP",
    "BA","BAC","BK","BKNG","BLK","BMY","BRK-B","C","CAT","CHTR","CL","CMCSA","COF",
    "COP","COST","CRM","CSCO","CVS","CVX","DHR","DIS","DOW","DUK","EMR","F","FDX",
    "GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","JNJ",
    "JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","MET","META","MMM",
    "MO","MRK","MS","MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE","PG","PM",
    "PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA","TXN",
    "UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM",
]

START = "2024-01-01"
END = "2025-05-31"


# ── EDGAR ────────────────────────────────────────────────────────────────────

def get_cik_map() -> Dict[str, str]:
    """ticker (upper) -> zero-padded 10-digit CIK."""
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


def _parse_items(raw: str) -> List[str]:
    """EDGAR 'items' field -> list of bare item codes like ['2.02','9.01'].

    The field looks like 'Item 2.02: Results...,Item 9.01: Financial...' or
    just '2.02,9.01' depending on entity. Extract the X.YY codes robustly.
    """
    return re.findall(r"\d\.\d{2}", raw or "")


def is_informative(form: str, items: List[str]) -> bool:
    """Keep all 10-Q; keep an 8-K only if it has >=1 informative item."""
    if form == "10-Q":
        return True
    if form == "8-K":
        return any(it in INFORMATIVE_ITEMS for it in items)
    return False


def get_filings(cik: str) -> List[Dict]:
    """Recent 8-K / 10-Q filings for a CIK within [START, END].

    8-Ks are filtered to informative item codes (see INFORMATIVE_ITEMS); the
    8-K item list is attached so it can be written to the panel for later
    per-item analysis.
    """
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    items_col = recent.get("items", [""] * len(forms))
    out = []
    for form, date, accn, doc, raw_items in zip(
            forms, dates, accns, docs, items_col):
        if form not in FORMS or not (START <= date <= END):
            continue
        items = _parse_items(raw_items)
        if not is_informative(form, items):
            continue
        out.append({"form": form, "date": date,
                    "accession": accn.replace("-", ""), "doc": doc,
                    "items": "|".join(items)})
    return out


def _clean_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)             # strip tags
    text = re.sub(r"&[a-z]+;", " ", text)           # strip entities
    return re.sub(r"\s+", " ", text).strip()


# financial-content keywords used to confirm we grabbed a substantive document
_INFO_KEYWORDS = re.compile(
    r"revenue|net income|earnings|per share|eps|quarter|guidance|operating|"
    r"margin|sales|profit|loss|outlook|results|fiscal", re.IGNORECASE)


def looks_informative(text: str, min_len: int = 400,
                      min_keywords: int = 3) -> bool:
    """True if the text is long enough AND mentions enough financial terms.

    Filters out cover pages / boilerplate (which have legal language but no
    financial content) so we don't waste LLM calls or pollute the panel.
    """
    if len(text) < min_len:
        return False
    return len(set(_INFO_KEYWORDS.findall(text))) >= min_keywords


def find_press_release_doc(cik: str, accession: str) -> Optional[str]:
    """Look in the filing's index.json for an EX-99 press-release exhibit.

    Returns the document filename, or None if no EX-99.x is present.
    """
    idx_url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{accession}/index.json")
    try:
        r = requests.get(idx_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        items = r.json().get("directory", {}).get("item", [])
    except Exception:
        return None
    # prefer EX-99.1, then any EX-99.x; match on the 'type' field when present,
    # else on filename hints
    candidates = []
    for it in items:
        name = it.get("name", "")
        typ = (it.get("type") or "").upper()
        if not name.lower().endswith((".htm", ".html", ".txt")):
            continue
        score = 0
        if typ.startswith("EX-99.1") or "EX-99.1" in typ:
            score = 3
        elif typ.startswith("EX-99") or "EX-99" in typ:
            score = 2
        elif re.search(r"ex.?99", name, re.IGNORECASE):
            score = 1
        if score:
            candidates.append((score, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def fetch_filing_text(cik: str, accession: str, doc: str,
                      max_chars: int = 8000) -> Tuple[str, str]:
    """Fetch the most informative text for a filing.

    Strategy: try the EX-99 press-release exhibit first (where earnings detail
    lives); fall back to the primary document. Returns (text, source) where
    source is 'ex99' or 'primary'. Text is validity-checked by the caller.
    """
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"

    # 1) try press release exhibit
    pr_doc = find_press_release_doc(cik, accession)
    if pr_doc:
        try:
            r = requests.get(f"{base}/{pr_doc}", headers=HEADERS, timeout=30)
            r.raise_for_status()
            text = _clean_html(r.text)
            if looks_informative(text):
                return text[:max_chars], "ex99"
        except Exception:
            pass

    # 2) fall back to primary document
    try:
        r = requests.get(f"{base}/{doc}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        text = _clean_html(r.text)
        return text[:max_chars], "primary"
    except Exception:
        return "", "primary"


# ── LLM scoring ──────────────────────────────────────────────────────────────

SCORING_PROMPT = """You are a financial event analyst reading this filing the \
MOMENT it is published. You have NO knowledge of how the stock moved afterward, \
and you must NOT use any memory of this company's later results or price action.

Score the filing's likely IMPACT on the stock price, based only on the content
of the text — revenue/earnings direction, margin trend, guidance language,
management tone, and material events (restatements, executive departures, large
deals, buybacks, dividend changes). Judge it as bullish or bearish on its own
substance.

score scale (impact on price):
 +1.0 extremely bullish | +0.5 moderately bullish | +0.2 mildly bullish
  0.0 neutral / routine / no financial substance
 -0.2 mildly bearish | -0.5 moderately bearish | -1.0 extremely bearish

confidence: how strongly the TEXT supports your score
 (0.0 = little basis in text, 1.0 = text states it clearly)

event_type must be one of: earnings, insider, macro, policy, 8-K, 10-Q

Respond with ONE LINE of strict JSON, nothing else:
{{"event_type": "...", "score": 0.0, "confidence": 0.0, "summary": "max 12 words no commas"}}

FILING EXCERPT:
{excerpt}
"""


def parse_llm_response(raw: str) -> Optional[Tuple[str, float, float, str]]:
    """Extract (event_type, score, confidence, summary) from the reply.
    Pure function — unit-tested without any network."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        etype = str(obj["event_type"]).strip()
        score = max(-1.0, min(1.0, float(obj["score"])))        # clip [-1,1]
        conf = max(0.0, min(1.0, float(obj.get("confidence", 1.0))))  # clip [0,1]
        summary = str(obj.get("summary", "")).replace(",", " ").strip()
        return etype, score, conf, summary
    except (ValueError, KeyError, TypeError):
        return None


def make_client():
    from openai import OpenAI
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    if not key:
        sys.exit("No API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
    return OpenAI(api_key=key, base_url=base)


def score_text_llm(client, text: str,
                   model: Optional[str] = None
                   ) -> Optional[Tuple[str, float, float, str]]:
    model = model or os.getenv("LANGCHAIN_MODEL_NAME", "deepseek-chat")
    resp = client.chat.completions.create(
        model=model,
        temperature=0,                              # determinism
        messages=[{"role": "user",
                   "content": SCORING_PROMPT.format(excerpt=text)}],
    )
    return parse_llm_response(resp.choices[0].message.content)


# ── main ─────────────────────────────────────────────────────────────────────

def main(tickers_subset: Optional[List[str]] = None):
    CACHE.mkdir(exist_ok=True)
    TEXTS.mkdir(exist_ok=True)
    scored_path = CACHE / "scored.json"
    scored = json.loads(scored_path.read_text()) if scored_path.exists() else {}
    panel_path = CACHE / "events_panel.csv"
    new_file = not panel_path.exists()

    client = make_client()
    cik_map = get_cik_map()
    universe = tickers_subset if tickers_subset else SP100

    with open(panel_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "ticker", "event_type", "score", "confidence",
                        "accession", "summary", "form", "items",
                        "text_source", "text_len"])

        for i, ticker in enumerate(universe):
            cik = cik_map.get(ticker.replace("-", ".").upper()) or cik_map.get(ticker)
            if not cik:
                print(f"[skip] no CIK for {ticker}")
                continue
            try:
                filings = get_filings(cik)
            except Exception as e:
                print(f"[err] filings {ticker}: {e}")
                continue
            print(f"[{i+1}/{len(universe)}] {ticker}: {len(filings)} informative filings")

            for fl in filings:
                accn = fl["accession"]
                if accn in scored:
                    continue
                try:
                    text, source = fetch_filing_text(cik, accn, fl["doc"])
                    time.sleep(0.2)                 # SEC rate-limit courtesy
                    if not looks_informative(text):
                        scored[accn] = True         # mark seen, skip scoring
                        continue
                    (TEXTS / f"{accn}.txt").write_text(text)
                    parsed = score_text_llm(client, text)
                    if not parsed:
                        continue
                    etype, score, conf, summary = parsed
                    w.writerow([fl["date"], ticker, etype, score, conf, accn,
                                summary, fl["form"], fl["items"],
                                source, len(text)])
                    f.flush()
                    scored[accn] = True
                except Exception as e:
                    print(f"  [err] {ticker} {accn}: {e}")
            scored_path.write_text(json.dumps(scored))

    print(f"\nDone. Events -> {panel_path}")


# ── self-test (no network) ───────────────────────────────────────────────────

def _selftest():
    cases = [
        ('{"event_type":"earnings","score":0.8,"confidence":0.9,"summary":"Q1 revenue beat strongly"}',
         ("earnings", 0.8, 0.9)),
        ('blah {"event_type":"8-K","score":-1.7,"confidence":1.5,"summary":"fraud, restated"} x',
         ("8-K", -1.0, 1.0)),  # score+conf clipped, comma stripped
        ('{"event_type":"10-Q","score":0.1,"summary":"no confidence field"}',
         ("10-Q", 0.1, 1.0)),  # missing confidence defaults to 1.0
        ("not json at all", None),
    ]
    for raw, expect in cases:
        got = parse_llm_response(raw)
        if expect is None:
            assert got is None, got
        else:
            assert got[:3] == expect, (got, expect)
            assert "," not in got[3]
    print("parse_llm_response self-test: PASS")

    # item parsing + informative filter
    assert _parse_items("Item 2.02: Results,Item 9.01: Financial") == ["2.02", "9.01"]
    assert _parse_items("5.03") == ["5.03"]
    assert _parse_items("") == []
    assert is_informative("8-K", ["2.02", "9.01"]) is True   # has earnings item
    assert is_informative("8-K", ["7.01"]) is False          # Reg FD only -> noise
    assert is_informative("8-K", ["9.01"]) is False          # exhibits only -> noise
    assert is_informative("8-K", []) is False
    assert is_informative("10-Q", []) is True                # always keep 10-Q
    print("item-filter self-test: PASS")

    # validity gate
    assert looks_informative(
        ("Company reported quarterly revenue of $5B, net income up, EPS beat, "
         "raising full-year guidance and operating margin expanded. " * 6)) is True
    assert looks_informative("Check the box if this is a pre-commencement "
                             "communication pursuant to Rule 425.") is False
    assert looks_informative("short") is False
    print("validity-gate self-test: PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--tickers", type=str, default=None,
                        help="comma-separated subset, e.g. NVDA,AAPL,MSFT")
        args = ap.parse_args()
        subset = ([t.strip().upper() for t in args.tickers.split(",")]
                  if args.tickers else None)
        main(tickers_subset=subset)
