"""Statistical validation — mirrors Vibe-Trading's backtest/validation.py
(monte_carlo_test, bootstrap_sharpe_ci, walk_forward_analysis) but adapted to a
cross-sectional daily-return / IC series instead of round-trip TradeRecords.

Three independent checks:
  - Monte Carlo permutation: is the IC better than random signal-return pairing?
  - Bootstrap Sharpe CI: how stable is the long-short risk-adjusted return?
  - Walk-Forward: is the signal consistent across sequential time windows?
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _sharpe(returns: np.ndarray, bars_per_year: int = 252) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(bars_per_year))


def monte_carlo_ic_test(signal_panel: pd.DataFrame,
                        fwd_ret: pd.DataFrame,
                        ic_fn=None,
                        n_simulations: int = 1000,
                        min_names: int = 10,
                        seed: int = 42) -> Dict[str, Any]:
    """Permutation test on mean IC.

    Null: the signal carries no cross-sectional information — the observed mean
    IC is no better than pairing each date's signals with a random permutation
    of that date's forward returns.

    Fast path: rank signal and returns ONCE; each simulation just shuffles the
    return-rank row in place (a permutation of ranks == a permutation of the
    raw returns) and recomputes per-row correlation. ic_fn is accepted for
    backward-compat but unused on the fast path.
    """
    from signals import _row_rank, _row_corr  # reuse vectorized core

    common = signal_panel.index.intersection(fwd_ret.index)
    sig_rank = _row_rank(signal_panel.loc[common].to_numpy(dtype=float))
    ret_rank = _row_rank(fwd_ret.loc[common].to_numpy(dtype=float))

    actual = np.nanmean(_row_corr(sig_rank, ret_rank, min_names))
    rng = np.random.default_rng(seed)
    sims = np.empty(n_simulations)
    for s in range(n_simulations):
        shuffled = ret_rank.copy()
        for r in range(shuffled.shape[0]):
            row = shuffled[r]
            valid = ~np.isnan(row)
            if valid.sum() >= 2:
                row[valid] = rng.permutation(row[valid])
        sims[s] = np.nanmean(_row_corr(sig_rank, shuffled, min_names))

    count = int(np.sum(np.abs(sims) >= abs(actual)))
    return {
        "actual_mean_ic": round(float(actual), 4),
        "p_value": round((count + 1) / (n_simulations + 1), 4),
        "sim_ic_p95": round(float(np.nanpercentile(sims, 95)), 4),
        "sim_ic_p05": round(float(np.nanpercentile(sims, 5)), 4),
        "n_simulations": n_simulations,
    }


def bootstrap_sharpe_ci(daily_ret: pd.Series,
                        n_bootstrap: int = 1000,
                        confidence: float = 0.95,
                        bars_per_year: int = 252,
                        seed: int = 42) -> Dict[str, Any]:
    """Resample daily long-short returns to estimate Sharpe CI.
    Same method as the repo's bootstrap_sharpe_ci.
    """
    returns = daily_ret.dropna().values
    if len(returns) < 5:
        return {"error": "need at least 5 return observations"}
    observed = _sharpe(returns, bars_per_year)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(returns)
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        boots.append(_sharpe(sample, bars_per_year))
    boots = np.array(boots)
    lo = (1 - confidence) / 2 * 100
    hi = (1 + confidence) / 2 * 100
    return {
        "observed_sharpe": round(observed, 4),
        "ci_lower": round(float(np.percentile(boots, lo)), 4),
        "ci_upper": round(float(np.percentile(boots, hi)), 4),
        "median_sharpe": round(float(np.median(boots)), 4),
        "prob_positive": round(float((boots > 0).mean()), 4),
        "n_bootstrap": n_bootstrap,
    }


def walk_forward_ic(ic: pd.Series,
                    ls_ret: pd.Series,
                    n_windows: int = 5,
                    bars_per_year: int = 252) -> Dict[str, Any]:
    """Split the timeline into sequential windows; report per-window mean IC
    and long-short Sharpe, plus a consistency score (fraction of windows with
    positive mean IC). Mirrors repo walk_forward_analysis intent.
    """
    ic = ic.dropna().sort_index()
    if len(ic) < n_windows * 2:
        return {"error": f"need at least {n_windows * 2} IC obs"}
    idx = ic.index
    size = len(idx) // n_windows
    windows = []
    pos = 0
    for i in range(n_windows):
        start = i * size
        end = (i + 1) * size if i < n_windows - 1 else len(idx)
        win_ic = ic.iloc[start:end]
        win_dates = win_ic.index
        win_ls = ls_ret.reindex(win_dates).dropna()
        m_ic = float(win_ic.mean())
        if m_ic > 0:
            pos += 1
        windows.append({
            "window": i + 1,
            "start": str(win_ic.index[0].date()),
            "end": str(win_ic.index[-1].date()),
            "mean_ic": round(m_ic, 4),
            "ls_sharpe": round(_sharpe(win_ls.values, bars_per_year), 4),
        })
    return {
        "per_window": windows,
        "consistency": round(pos / n_windows, 4),
        "n_windows": n_windows,
    }
