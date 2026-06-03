"""EVENT STUDY — the right validation tool for a SMALL universe.

Cross-sectional IC needs many names per day. On ~10 tickers most days don't
have enough names, so IC is meaningless. Instead we ask the more fundamental
question directly:

  After an event, does the stock drift in the direction of its LLM score?

For each scored event we compute cumulative abnormal-ish return (raw CAR here,
market-adjustment optional) over horizons T+1..T+H, then bucket events into
POSITIVE / NEUTRAL / NEGATIVE by score and compare mean drift per bucket.

If positive-scored events drift up and negative-scored events drift down, the
signal carries information — regardless of universe size. This is the check to
pass BEFORE spending time/quota on a full cross-sectional run.

Look-ahead guard: CAR is measured strictly AFTER the filing date (T+1 onward),
and the event date is the filing's knowable date.

Usage:
  python event_study.py                 # uses cache/events_panel.csv + prices
  python event_study.py --horizons 1,3,5,10
  python event_study.py --pos 0.3 --neg -0.3   # bucket thresholds
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
START = "2024-01-01"
END = "2025-05-31"


def load_prices(tickers: List[str]) -> Dict[str, pd.Series]:
    pq, pk = CACHE / "prices.parquet", CACHE / "prices.pkl"
    if pq.exists():
        wide = pd.read_parquet(pq)
    elif pk.exists():
        wide = pd.read_pickle(pk)
    else:
        import yfinance as yf
        raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                          progress=False)
        wide = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        CACHE.mkdir(exist_ok=True)
        try:
            wide.to_parquet(pq)
        except Exception:
            wide.to_pickle(pk)
    out = {}
    for t in wide.columns:
        s = wide[t].dropna()
        s.index = pd.DatetimeIndex(s.index)
        if len(s) > 30:
            out[t] = s
    return out


def load_market() -> Optional[pd.Series]:
    """SPY close series, cached separately (the main price cache may lack it)."""
    f = CACHE / "spy.pkl"
    if f.exists():
        return pd.read_pickle(f)
    try:
        import yfinance as yf
        raw = yf.download("SPY", start=START, end=END, auto_adjust=True,
                          progress=False)
        s = (raw["Close"] if "Close" in raw.columns else raw).squeeze()
        s.index = pd.DatetimeIndex(s.index)
        s = s.dropna()
        CACHE.mkdir(exist_ok=True)
        s.to_pickle(f)
        return s
    except Exception as e:
        print(f"[warn] could not load SPY: {e}")
        return None


def car_after(close: pd.Series, event_date: pd.Timestamp, h: int,
              mkt: Optional[pd.Series] = None):
    """Return from entry (first close after event) held h trading days.

    Entry at after[0] (= T+1, the first knowable close post-filing).
    Exit at after[h] (h trading days later). So h=1 is a genuine one-day
    forward return entry->next day, not a zero.

    If mkt (a benchmark close series, e.g. SPY) is given, returns the ABNORMAL
    return = stock return - market return over the same window, stripping out
    beta-1 market drift so what's left is event-specific. Returns None if
    insufficient forward data.
    """
    idx = close.index
    after = idx[idx > event_date]
    if len(after) < h + 1:
        return None
    entry = close.loc[after[0]]
    exit_ = close.loc[after[h]]
    if entry == 0:
        return None
    stock_ret = float(exit_ / entry - 1.0)
    if mkt is None:
        return stock_ret
    # market return over the same calendar window
    try:
        m_entry = mkt.loc[mkt.index[mkt.index >= after[0]][0]]
        m_exit = mkt.loc[mkt.index[mkt.index >= after[h]][0]]
        if m_entry == 0:
            return stock_ret
        mkt_ret = float(m_exit / m_entry - 1.0)
        return stock_ret - mkt_ret
    except (IndexError, KeyError):
        return stock_ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="1,3,5,10")
    ap.add_argument("--pos", type=float, default=0.2)
    ap.add_argument("--neg", type=float, default=-0.2)
    ap.add_argument("--form", type=str, default=None,
                    help="restrict to a form, e.g. 8-K or 10-Q")
    ap.add_argument("--item", type=str, default=None,
                    help="restrict to events whose items contain this code, e.g. 2.02")
    ap.add_argument("--market", action="store_true",
                    help="use market-adjusted abnormal returns (stock - SPY)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop events with confidence below this (0-1)")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    events = pd.read_csv(CACHE / "events_panel.csv")
    events["date"] = pd.to_datetime(events["date"])
    if args.form and "form" in events.columns:
        events = events[events["form"] == args.form]
    if args.item and "items" in events.columns:
        events = events[events["items"].fillna("").str.contains(args.item)]
    if args.min_conf > 0 and "confidence" in events.columns:
        before = len(events)
        events = events[events["confidence"].fillna(1.0) >= args.min_conf]
        print(f"[filter] confidence>={args.min_conf}: {before} -> {len(events)}")

    tickers = sorted(events["ticker"].unique())
    prices = load_prices(tickers)

    mkt = None
    if args.market:
        mkt = load_market()
        if mkt is None:
            print("[warn] SPY not available; falling back to raw returns")

    rows = []
    for _, e in events.iterrows():
        s = prices.get(e["ticker"])
        if s is None:
            continue
        close = s["close"] if isinstance(s, pd.DataFrame) else s
        rec = {"score": e["score"]}
        for h in horizons:
            rec[f"car_{h}"] = car_after(close, e["date"], h, mkt=mkt)
        rows.append(rec)
    df = pd.DataFrame(rows)

    def bucket(x):
        if x >= args.pos:
            return "POS"
        if x <= args.neg:
            return "NEG"
        return "NEU"
    df["bucket"] = df["score"].apply(bucket)

    print(f"\nEvents: {len(df)}  |  buckets: {df['bucket'].value_counts().to_dict()}")
    print(f"Score thresholds: POS>={args.pos}  NEG<={args.neg}\n")

    print(f"{'horizon':>8} | {'POS mean':>10} {'NEG mean':>10} "
          f"{'spread':>10} {'POS hit':>8} {'NEG hit':>8} {'n':>6}")
    print("-" * 70)
    for h in horizons:
        col = f"car_{h}"
        d = df[["bucket", col]].dropna()
        pos = d[d["bucket"] == "POS"][col]
        neg = d[d["bucket"] == "NEG"][col]
        if len(pos) == 0 or len(neg) == 0:
            print(f"{h:>8} | insufficient data in one bucket")
            continue
        spread = pos.mean() - neg.mean()
        pos_hit = (pos > 0).mean()      # positive events that actually went up
        neg_hit = (neg < 0).mean()      # negative events that actually went down
        print(f"{h:>8} | {pos.mean():>10.4f} {neg.mean():>10.4f} "
              f"{spread:>10.4f} {pos_hit:>8.2%} {neg_hit:>8.2%} {len(d):>6}")

    print("\nRead: a POSITIVE 'spread' (POS drift above NEG drift) means the "
          "score has directional information. Hit rates >50% reinforce it.")
    # correlation of raw score with each horizon's CAR (Spearman)
    print("\nScore vs CAR rank correlation (Spearman):")
    for h in horizons:
        col = f"car_{h}"
        d = df[["score", col]].dropna()
        if len(d) > 5:
            rho = d["score"].corr(d[col], method="spearman")
            print(f"  T+{h}: rho={rho:+.4f}  (n={len(d)})")


if __name__ == "__main__":
    main()
