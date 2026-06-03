"""SINGLE-FACTOR BACKTEST FRAMEWORK — the sentiment factor, done properly.

Goes beyond IC to a full factor quality assessment, with strict train/test
discipline so parameter choices can't leak into the reported result.

Pipeline:
  1. Build the pure sentiment factor (confidence-weighted, time-decayed event
     score) as a date x ticker panel. No technical signal in the factor.
  2. TRAIN window: grid-search a small set of parameters (decay lambda,
     confidence floor, quantile count) and pick the best by IC-IR on TRAIN only.
  3. TEST window: evaluate ONCE with the chosen parameters and report a full
     quality card:
       - predictive power: IC mean / IC-IR / t-stat / hit rate
       - quintile portfolios: per-quantile mean fwd return (monotonicity),
         long-short annualized return / vol / Sharpe / max drawdown / win rate
       - turnover and COST-ADJUSTED Sharpe (single-side bps)
       - coverage: non-zero signal names per day
  4. Statistical validation (Monte Carlo / bootstrap / walk-forward).
  5. run_card.{json,md} split into TRAIN (choices) and TEST (conclusions).

The meaningful control here is NOT a momentum baseline — on large-cap daily
horizons both sentiment and momentum are near-zero, so comparing them is
uninformative. The meaningful control is the FinBERT contamination check
(finbert_baseline.py): LLM vs a fixed classifier on identical text, which tests
look-ahead leakage rather than whether the factor beats momentum.

Honest notes baked in: limited sample -> grid is intentionally tiny; the report
flags degrees of freedom and coverage. This framework is built to evaluate a
factor rigorously, not to make a weak factor look strong.

Usage:
  python factor_eval.py
  python factor_eval.py --train-end 2024-12-31 --horizon 5
  python factor_eval.py --cost-bps 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import signals as S
import validation as V

CACHE = Path(__file__).resolve().parent / "cache"
OUT = Path(__file__).resolve().parent / "out"
START, END = "2024-01-01", "2025-05-31"
TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(tickers: List[str]) -> Dict[str, pd.DataFrame]:
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
            df = pd.DataFrame({"close": s})
            df.index = pd.DatetimeIndex(df.index)
            out[t] = df
    return out


def build_sentiment_panel(events: pd.DataFrame,
                          prices: Dict[str, pd.DataFrame],
                          decay_lambda: float,
                          conf_floor: float) -> pd.DataFrame:
    """Pure sentiment factor: confidence-weighted, time-decayed event score.

    No technical component. Events below the confidence floor are dropped; the
    surviving score is multiplied by confidence (soft weighting) before decay.
    """
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    if "confidence" in ev.columns:
        ev = ev[ev["confidence"].fillna(1.0) >= conf_floor]
        ev["score"] = ev["score"] * ev["confidence"].fillna(1.0)  # soft weight
    panel = {}
    for ticker, df in prices.items():
        dates = pd.DatetimeIndex(df.sort_index().index)
        e = ev[ev["ticker"] == ticker]
        panel[ticker] = S.compute_event_signal(e, dates, decay_lambda=decay_lambda)
    return pd.DataFrame(panel)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio analytics
# ─────────────────────────────────────────────────────────────────────────────

def max_drawdown(equity: pd.Series) -> float:
    roll = equity.cummax()
    return float((equity / roll - 1.0).min())


def quantile_analytics(signal: pd.DataFrame, fwd: pd.DataFrame,
                       n_q: int, cost_bps: float,
                       min_names: int = 10) -> Dict:
    """Quintile long-short analytics with turnover and cost adjustment."""
    common = signal.index.intersection(fwd.index)
    q_returns = {q: [] for q in range(n_q)}          # per-quantile mean fwd ret
    ls, dates = [], []
    prev_long, prev_short = set(), set()
    turnover = []
    for d in common:
        s = signal.loc[d]
        r = fwd.loc[d]
        j = pd.concat([s, r], axis=1, keys=["s", "r"]).dropna()
        if len(j) < min_names or j["s"].std() == 0:
            continue
        try:
            q = pd.qcut(j["s"].rank(method="first"), n_q, labels=False,
                        duplicates="drop")
        except ValueError:
            continue
        for qi in range(n_q):
            vals = j["r"][q == qi]
            if len(vals):
                q_returns[qi].append(vals.mean())
        long_names = set(j.index[q == q.max()])
        short_names = set(j.index[q == q.min()])
        ls.append(j["r"][q == q.max()].mean() - j["r"][q == q.min()].mean())
        dates.append(d)
        # turnover: fraction of the long+short book that changed vs prior day
        if prev_long or prev_short:
            changed = (len(long_names ^ prev_long) + len(short_names ^ prev_short))
            base = (len(long_names) + len(short_names)) or 1
            turnover.append(changed / (2 * base))
        prev_long, prev_short = long_names, short_names

    ls = pd.Series(ls, index=pd.DatetimeIndex(dates)).sort_index()
    if len(ls) < 5:
        return {"error": "insufficient long-short observations"}
    avg_turnover = float(np.mean(turnover)) if turnover else 0.0
    # cost per day = turnover * 2 sides * cost_bps
    daily_cost = avg_turnover * 2 * (cost_bps / 1e4)
    ls_net = ls - daily_cost

    def sharpe(x):
        x = x.dropna()
        return float(x.mean() / x.std() * np.sqrt(TRADING_DAYS)) if x.std() else 0.0

    eq = (1 + ls.fillna(0)).cumprod()
    eq_net = (1 + ls_net.fillna(0)).cumprod()
    return {
        "quantile_mean_fwd_ret": {f"Q{qi+1}": round(float(np.mean(v)), 5)
                                  for qi, v in q_returns.items() if v},
        "ls_ann_return": round(float(ls.mean() * TRADING_DAYS), 4),
        "ls_ann_vol": round(float(ls.std() * np.sqrt(TRADING_DAYS)), 4),
        "ls_sharpe_gross": round(sharpe(ls), 4),
        "ls_sharpe_net": round(sharpe(ls_net), 4),
        "ls_max_drawdown": round(max_drawdown(eq), 4),
        "ls_max_drawdown_net": round(max_drawdown(eq_net), 4),
        "ls_win_rate": round(float((ls > 0).mean()), 4),
        "avg_daily_turnover": round(avg_turnover, 4),
        "cost_bps_per_side": cost_bps,
        "n_days": len(ls),
        "_ls_series": ls,           # for validation, stripped before JSON
    }


def coverage(signal: pd.DataFrame) -> Dict:
    nonzero = (signal != 0).sum(axis=1)
    return {"mean_nonzero_per_day": round(float(nonzero.mean()), 2),
            "max_nonzero_per_day": int(nonzero.max()),
            "pct_days_any_signal": round(float((nonzero > 0).mean()), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(signal: pd.DataFrame, fwd: pd.DataFrame,
             n_q: int, cost_bps: float) -> Dict:
    ic = S.cross_sectional_ic(signal, fwd, min_names=10)
    card = {"ic": S.ic_summary(ic), "coverage": coverage(signal)}
    qa = quantile_analytics(signal, fwd, n_q, cost_bps)
    card["portfolio"] = {k: v for k, v in qa.items() if not k.startswith("_")}
    card["_ic_series"] = ic
    card["_ls_series"] = qa.get("_ls_series")
    return card


def select_params_on_train(events, prices, train_dates, horizon,
                           grid) -> Tuple[Dict, List]:
    """Grid-search params; pick best by TRAIN IC-IR. Returns (best, log)."""
    log = []
    best, best_ir = None, -np.inf
    for lam, cf, nq in product(grid["lambda"], grid["conf"], grid["nq"]):
        panel = build_sentiment_panel(events, prices, lam, cf)
        panel = panel.loc[panel.index.intersection(train_dates)]
        fwd = S.forward_returns(prices, horizon=horizon)
        fwd = fwd.loc[fwd.index.intersection(train_dates)]
        ic = S.cross_sectional_ic(panel, fwd, min_names=10)
        summ = S.ic_summary(ic)
        ir = summ.get("ic_ir") or -np.inf
        log.append({"lambda": lam, "conf": cf, "nq": nq,
                    "train_ic_ir": summ.get("ic_ir"),
                    "train_mean_ic": summ.get("mean_ic")})
        if ir > best_ir:
            best_ir, best = ir, {"lambda": lam, "conf": cf, "nq": nq}
    return best, log


def strip_private(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# run_card
# ─────────────────────────────────────────────────────────────────────────────

def write_run_card(payload: Dict):
    OUT.mkdir(exist_ok=True)
    h = hashlib.sha256(json.dumps(payload["config"], sort_keys=True).encode()
                       ).hexdigest()[:12]
    payload["config_hash"] = h
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "factor_run_card.json").write_text(
        json.dumps(payload, indent=2, default=str))

    L = [f"# Sentiment Factor — Run Card", "",
         f"_Generated {payload['generated_utc']} · config `{h}`_", "",
         "## Config", ""]
    for k, v in payload["config"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## TRAIN — parameter selection", "",
          f"- chosen params: {payload['chosen_params']}",
          f"- grid size searched: {payload['grid_size']} combinations "
          f"(degrees of freedom — kept small given sample size)", ""]
    for sec_name, key in [("TEST — Sentiment factor", "test_sentiment")]:
        if key not in payload:
            continue
        c = payload[key]
        L += [f"## {sec_name}", "", "### Predictive power"]
        for k, v in c["ic"].items():
            L.append(f"- {k}: {v}")
        L += ["", "### Quintile portfolio"]
        port = c["portfolio"]
        if "quantile_mean_fwd_ret" in port:
            L.append(f"- per-quantile mean fwd return: {port['quantile_mean_fwd_ret']}")
        for k in ("ls_ann_return", "ls_sharpe_gross", "ls_sharpe_net",
                  "ls_max_drawdown", "ls_win_rate", "avg_daily_turnover", "n_days"):
            if k in port:
                L.append(f"- {k}: {port[k]}")
        L += ["", "### Coverage"]
        for k, v in c["coverage"].items():
            L.append(f"- {k}: {v}")
        L.append("")
    L += ["## Note on controls", "",
          "A momentum baseline is intentionally omitted: on large-cap daily "
          "horizons both this factor and momentum are near-zero, so a "
          "sentiment-vs-momentum comparison is uninformative. The meaningful "
          "control is the FinBERT contamination check (run_backtest.py / "
          "finbert_baseline.py) — LLM vs a fixed classifier on identical text, "
          "testing look-ahead leakage rather than factor superiority.", ""]
    if "validation" in payload:
        L += ["## Statistical validation (TEST, sentiment factor)", ""]
        for block, d in payload["validation"].items():
            L.append(f"**{block}**")
            for k, v in d.items():
                if not str(k).startswith("_"):
                    L.append(f"- {k}: {v}")
            L.append("")
    if payload.get("warnings"):
        L += ["## Warnings", ""] + [f"- ⚠ {w}" for w in payload["warnings"]]
    (OUT / "factor_run_card.md").write_text("\n".join(L))
    print(f"\nrun_card -> {OUT/'factor_run_card.md'} & .json")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=str, default="2024-12-31")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--item", type=str, default=None)
    args = ap.parse_args()

    events = pd.read_csv(CACHE / "events_panel.csv")
    if args.item and "items" in events.columns:
        events = events[events["items"].fillna("").str.contains(args.item)]
    tickers = sorted(events["ticker"].unique())
    prices = load_prices(tickers)

    # train/test split by date
    all_dates = pd.DatetimeIndex(sorted(
        set().union(*[df.index for df in prices.values()])))
    train_end = pd.Timestamp(args.train_end)
    train_dates = all_dates[all_dates <= train_end]
    test_dates = all_dates[all_dates > train_end]

    # tiny grid (sample is small — keep DoF low)
    grid = {"lambda": [0.05, 0.1, 0.3], "conf": [0.0, 0.5], "nq": [5]}
    grid_size = len(grid["lambda"]) * len(grid["conf"]) * len(grid["nq"])
    best, train_log = select_params_on_train(
        events, prices, train_dates, args.horizon, grid)

    # build factor with chosen params, evaluate ONCE on test
    sent = build_sentiment_panel(events, prices, best["lambda"], best["conf"])
    fwd = S.forward_returns(prices, horizon=args.horizon)
    sent_test = sent.loc[sent.index.intersection(test_dates)]
    fwd_test = fwd.loc[fwd.index.intersection(test_dates)]
    test_sent = evaluate(sent_test, fwd_test, best["nq"], args.cost_bps)

    payload = {
        "config": {
            "universe_tickers": len(tickers),
            "window": f"{START}..{END}",
            "train_end": args.train_end, "horizon": args.horizon,
            "cost_bps_per_side": args.cost_bps,
            "factor": "pure sentiment (confidence-weighted, time-decayed)",
            "item_filter": args.item,
        },
        "chosen_params": best, "grid_size": grid_size,
        "train_grid_log": train_log,
        "test_sentiment": strip_private(test_sent),
        "warnings": [],
    }

    # validation on test sentiment factor
    ls = test_sent.get("_ls_series")
    payload["validation"] = {
        "monte_carlo": V.monte_carlo_ic_test(sent_test, fwd_test, n_simulations=1000),
        "bootstrap_sharpe": (V.bootstrap_sharpe_ci(ls) if ls is not None
                             else {"error": "no ls series"}),
        "walk_forward": (V.walk_forward_ic(test_sent["_ic_series"], ls, n_windows=4)
                         if ls is not None else {}),
    }

    # warnings
    n_ic = test_sent["ic"].get("n", 0)
    if n_ic < 60:
        payload["warnings"].append(
            f"Only {n_ic} test IC days — low statistical power.")
    if payload["validation"]["monte_carlo"]["p_value"] > 0.05:
        payload["warnings"].append(
            "Sentiment factor IC not significant vs random on test (p>0.05).")
    if test_sent["coverage"]["mean_nonzero_per_day"] < 10:
        payload["warnings"].append(
            "Sparse coverage (<10 names/day) — quantile portfolios noisy.")

    write_run_card(payload)
    print("\n=== TEST sentiment IC ===", test_sent["ic"])
    print("=== TEST sentiment portfolio ===",
          {k: test_sent["portfolio"].get(k) for k in
           ("ls_sharpe_gross", "ls_sharpe_net", "ls_max_drawdown",
            "avg_daily_turnover")})
    print("=== MC ===", payload["validation"]["monte_carlo"])


if __name__ == "__main__":
    main()
