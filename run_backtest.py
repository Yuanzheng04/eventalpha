"""STAGE 2 (fast, no LLM): cross-sectional backtest + validation + run_card.

Reads the cached event panel(s) from stage 1, pulls OHLCV, builds the
cross-sectional signal, and produces the resume-bearing artifacts:

  - cross-sectional Information Coefficient (mean IC, IC-IR, t-stat, hit rate)
  - quintile long-short daily returns + equity curve
  - Monte Carlo permutation p-value, Bootstrap Sharpe CI, Walk-Forward
  - FinBERT contamination check (LLM IC vs FinBERT IC on identical text)
  - run_card.json + run_card.md

Re-runnable in seconds: tune alpha / decay / quantiles without re-scoring.

Usage:
  python run_backtest.py                 # uses cache/events_panel.csv + yfinance
  python run_backtest.py --alpha 0.5
  python run_backtest.py --no-finbert    # skip contamination check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import signals as S
import validation as V

CACHE = Path(__file__).resolve().parent / "cache"
OUT = Path(__file__).resolve().parent / "out"
START = "2024-01-01"
END = "2025-05-31"


def load_prices(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """OHLCV via yfinance, cached on first pull (parquet if available, else pickle)."""
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
        if len(s) > 60:
            out[t] = pd.DataFrame({"close": s})
            out[t].index = pd.DatetimeIndex(out[t].index)
    return out


def run_pipeline(events: pd.DataFrame, prices: Dict[str, pd.DataFrame],
                 alpha: float, horizon: int = 1) -> Dict:
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    panel = S.build_signal_panel(prices, events, alpha=alpha, typed_decay=True)
    fwd = S.forward_returns(prices, horizon=horizon)
    ic = S.cross_sectional_ic(panel, fwd, min_names=10)
    ls = S.quantile_long_short(panel, fwd, n_quantiles=5, min_names=10)
    eq = S.equity_curve_from_returns(ls)
    return {"panel": panel, "fwd": fwd, "ic": ic, "ls": ls, "eq": eq}


def write_run_card(metrics: Dict, config: Dict, warnings: List[str]):
    OUT.mkdir(exist_ok=True)
    cfg_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    card = {
        "schema_version": "0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_hash": cfg_hash,
        "metrics": metrics,
        "warnings": warnings,
    }
    (OUT / "run_card.json").write_text(json.dumps(card, indent=2, default=str))

    md = [f"# Event-Driven Cross-Sectional Signal — Run Card",
          f"\n_Generated {card['generated_utc']} · config `{cfg_hash}`_\n",
          f"## Config\n",
          f"- Universe: {config['universe']} ({config['n_tickers']} names)",
          f"- Window: {config['start']} → {config['end']}",
          f"- Event source: {config['event_source']}",
          f"- alpha (tech weight): {config['alpha']} · typed decay: {config['typed_decay']}",
          f"- Signal lag: T+1 (event_date ≤ trade_date enforced)\n",
          f"## Information Coefficient (the headline result)\n"]
    for k, v in metrics["ic"].items():
        md.append(f"- {k}: {v}")
    md.append("\n## Long-Short Quintile Portfolio\n")
    for k, v in metrics["long_short"].items():
        md.append(f"- {k}: {v}")
    md.append("\n## Statistical Validation\n")
    md.append(f"**Monte Carlo permutation** (signal vs random pairing)")
    for k, v in metrics["monte_carlo"].items():
        md.append(f"- {k}: {v}")
    md.append(f"\n**Bootstrap Sharpe CI** (long-short)")
    for k, v in metrics["bootstrap"].items():
        md.append(f"- {k}: {v}")
    md.append(f"\n**Walk-Forward** (consistency across {metrics['walk_forward'].get('n_windows','?')} windows)")
    md.append(f"- consistency: {metrics['walk_forward'].get('consistency')}")
    for win in metrics["walk_forward"].get("per_window", []):
        md.append(f"  - W{win['window']} {win['start']}→{win['end']}: "
                  f"IC={win['mean_ic']} LS_Sharpe={win['ls_sharpe']}")
    if "contamination" in metrics:
        c = metrics["contamination"]
        md.append("\n## Look-Ahead Contamination Check (LLM vs FinBERT)\n")
        md.append(f"- LLM mean IC: {c['llm_mean_ic']}")
        md.append(f"- FinBERT mean IC: {c['finbert_mean_ic']}")
        md.append(f"- IC gap (LLM − FinBERT): {c['ic_gap']}")
        md.append(f"- interpretation: {c['interpretation']}")
    if warnings:
        md.append("\n## Warnings\n")
        for wn in warnings:
            md.append(f"- ⚠ {wn}")
    (OUT / "run_card.md").write_text("\n".join(md))
    print(f"\nrun_card -> {OUT/'run_card.md'}  &  run_card.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--no-finbert", action="store_true")
    ap.add_argument("--event-type", type=str, default=None,
                    help="filter to one event_type, e.g. earnings")
    ap.add_argument("--item", type=str, default=None,
                    help="filter to events whose items contain this code, e.g. 2.02")
    ap.add_argument("--horizon", type=int, default=1,
                    help="forward return horizon in trading days")
    args = ap.parse_args()

    events = pd.read_csv(CACHE / "events_panel.csv")
    if args.event_type:
        before = len(events)
        events = events[events["event_type"] == args.event_type]
        print(f"[filter] event_type={args.event_type}: {before} -> {len(events)}")
    if args.item and "items" in events.columns:
        before = len(events)
        events = events[events["items"].fillna("").str.contains(args.item)]
        print(f"[filter] item~{args.item}: {before} -> {len(events)}")
    tickers = sorted(events["ticker"].unique())
    prices = load_prices(tickers)

    r = run_pipeline(events, prices, alpha=args.alpha, horizon=args.horizon)
    ic_summ = S.ic_summary(r["ic"])
    ls = r["ls"]
    eq = r["eq"]

    metrics = {
        "ic": ic_summ,
        "long_short": {
            "total_return": round(float(eq.iloc[-1] / eq.iloc[0] - 1), 4) if len(eq) else None,
            "ann_sharpe": round(float(ls.mean() / ls.std() * np.sqrt(252)), 4) if ls.std() else None,
            "n_days": int(ls.notna().sum()),
        },
        "monte_carlo": V.monte_carlo_ic_test(r["panel"], r["fwd"],
                                             n_simulations=1000),
        "bootstrap": V.bootstrap_sharpe_ci(ls),
        "walk_forward": V.walk_forward_ic(r["ic"], ls, n_windows=5),
    }

    warnings = []
    if ic_summ.get("n", 0) < 60:
        warnings.append("Few IC observations (<60 days) — low statistical power.")
    if metrics["monte_carlo"]["p_value"] > 0.05:
        warnings.append("Monte Carlo p-value > 0.05 — signal NOT significant vs random.")

    # contamination check
    fb_path = CACHE / "events_panel_finbert.csv"
    if not args.no_finbert and fb_path.exists():
        fb = pd.read_csv(fb_path)
        if args.event_type:
            fb = fb[fb["event_type"] == args.event_type]
        if args.item and "items" in fb.columns:
            fb = fb[fb["items"].fillna("").str.contains(args.item)]
        rb = run_pipeline(fb, prices, alpha=args.alpha, horizon=args.horizon)
        llm_ic = ic_summ.get("mean_ic", 0.0)
        fb_ic = S.ic_summary(rb["ic"]).get("mean_ic", 0.0)
        gap = round(llm_ic - fb_ic, 4)
        if abs(fb_ic) < 1e-6:
            interp = "FinBERT IC ~0; inconclusive."
        elif llm_ic > 2 * abs(fb_ic) and llm_ic > 0.03:
            interp = "LLM IC >> FinBERT — POSSIBLE hindsight leakage, flag it."
        else:
            interp = "LLM IC comparable to FinBERT — leakage limited."
        metrics["contamination"] = {
            "llm_mean_ic": llm_ic, "finbert_mean_ic": fb_ic,
            "ic_gap": gap, "interpretation": interp,
        }
    elif not args.no_finbert:
        warnings.append("No FinBERT panel found — run finbert_baseline.py "
                        "for the contamination check.")

    config = {
        "universe": "S&P 100", "n_tickers": len(tickers),
        "start": START, "end": END,
        "event_source": "EDGAR 8-K/10-Q + LLM point-in-time scoring",
        "alpha": args.alpha, "typed_decay": True,
    }
    write_run_card(metrics, config, warnings)
    print("\n=== SUMMARY ===")
    print("IC:", ic_summ)
    print("MC:", metrics["monte_carlo"])
    if "contamination" in metrics:
        print("Contamination:", metrics["contamination"])


if __name__ == "__main__":
    main()
