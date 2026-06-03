"""Signal construction + cross-sectional evaluation engine.

Stage-2 numerical core. No LLM, no network — pure pandas/numpy, so it is fast
and can be re-run endlessly while tuning lambda / alpha / quantiles.

compute_event_signal and combine_signals are copied VERBATIM from
Vibe-Trading's event-driven SKILL.md so behaviour matches the local skill.

Everything below the divider is the cross-sectional layer this project adds:
information coefficient, quintile long/short portfolios, and look-ahead-safe
T+1 alignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Copied verbatim from agent/src/skills/event-driven/SKILL.md
# ─────────────────────────────────────────────────────────────────────────────

# Per-event-type decay (information half-life). Hand-set, NOT fit from data —
# this is a known limitation to disclose, not hide.
DECAY_BY_TYPE = {
    "earnings": 0.30,   # priced within ~3 days
    "sentiment": 0.40,  # noise, fades fast
    "insider": 0.10,    # medium-term
    "macro": 0.05,      # regime shift, persists
    "policy": 0.03,     # structural, long-lasting
    "8-K": 0.20,        # generic material event
    "10-Q": 0.15,       # quarterly report
    "default": 0.10,
}


def compute_event_signal(event_df: pd.DataFrame, dates: pd.DatetimeIndex,
                         decay_lambda: float = 0.1,
                         min_score_threshold: float = 0.2,
                         event_lookback: int = 30) -> pd.Series:
    """Event-driven signal with exponential time decay. (verbatim from skill)"""
    event_df = event_df[event_df["score"].abs() >= min_score_threshold].copy()
    event_df["date"] = pd.to_datetime(event_df["date"])

    signal = pd.Series(0.0, index=dates)

    for trade_date in dates:
        # Only events knowable on or before trade_date — the temporal
        # look-ahead guard. event_date <= trade_date, strictly.
        mask = (event_df["date"] <= trade_date) & \
               (event_df["date"] >= trade_date - pd.Timedelta(days=event_lookback))
        relevant = event_df[mask]
        if relevant.empty:
            continue
        days_since = (trade_date - relevant["date"]).dt.days.values
        scores = relevant["score"].values
        decayed = scores * np.exp(-decay_lambda * days_since)
        signal[trade_date] = np.clip(decayed.sum(), -1.0, 1.0)

    return signal


def compute_event_signal_typed(event_df: pd.DataFrame, dates: pd.DatetimeIndex,
                               min_score_threshold: float = 0.2,
                               event_lookback: int = 45) -> pd.Series:
    """Same as compute_event_signal but uses a per-event-type decay rate.

    Splits events by event_type and applies DECAY_BY_TYPE[type], then sums.
    This is the 'different lambda per event type' refinement discussed for the
    SKILL.md decay table.
    """
    out = pd.Series(0.0, index=dates)
    if event_df.empty:
        return out
    for etype, grp in event_df.groupby("event_type"):
        lam = DECAY_BY_TYPE.get(str(etype), DECAY_BY_TYPE["default"])
        out = out + compute_event_signal(
            grp, dates, decay_lambda=lam,
            min_score_threshold=min_score_threshold,
            event_lookback=event_lookback,
        )
    return out.clip(-1.0, 1.0)


def combine_signals(tech_signal: pd.Series, event_signal: pd.Series,
                    alpha: float = 0.6) -> pd.Series:
    """Weighted combine: alpha*tech + (1-alpha)*event. (verbatim from skill)"""
    combined = alpha * tech_signal + (1 - alpha) * event_signal
    return combined.clip(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional layer (added by this project)
# ─────────────────────────────────────────────────────────────────────────────


def technical_signal(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """Simple normalized momentum: tanh of (fast EMA / slow EMA - 1) scaled.

    Range ~[-1, 1]. Deterministic, no look-ahead (uses only past closes).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    raw = (ema_fast / ema_slow) - 1.0
    return np.tanh(raw * 20.0).clip(-1.0, 1.0)


def build_signal_panel(
    prices: Dict[str, pd.DataFrame],
    events: pd.DataFrame,
    alpha: float = 0.6,
    typed_decay: bool = True,
) -> pd.DataFrame:
    """Build a date x ticker panel of combined signals.

    Args:
        prices: {ticker -> OHLCV df indexed by date, must have 'close'}.
        events: long df with columns [date, ticker, event_type, score].
        alpha: tech weight in combine_signals.
        typed_decay: use per-event-type decay if True.

    Returns:
        DataFrame indexed by date, columns = tickers, values = combined signal.
    """
    panel = {}
    for ticker, df in prices.items():
        df = df.sort_index()
        dates = pd.DatetimeIndex(df.index)
        tech = technical_signal(df["close"])
        ev = events[events["ticker"] == ticker]
        if typed_decay:
            evsig = compute_event_signal_typed(ev, dates)
        else:
            evsig = compute_event_signal(ev, dates)
        panel[ticker] = combine_signals(tech, evsig, alpha=alpha)
    return pd.DataFrame(panel)


def forward_returns(prices: Dict[str, pd.DataFrame], horizon: int = 1) -> pd.DataFrame:
    """Forward N-day close-to-close return per ticker, date x ticker panel.

    fwd_ret[t] = close[t+horizon]/close[t] - 1. Signal known at close t,
    position held t -> t+horizon. The forward shift is the look-ahead guard.
    """
    out = {}
    for ticker, df in prices.items():
        df = df.sort_index()
        out[ticker] = (df["close"].shift(-horizon) / df["close"] - 1.0)
    return pd.DataFrame(out)


def _row_rank(mat: np.ndarray) -> np.ndarray:
    """NaN-aware rank along axis=1. NaNs stay NaN; valid entries get 0..k-1."""
    out = np.full(mat.shape, np.nan, dtype=float)
    for r in range(mat.shape[0]):
        row = mat[r]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        out[r, valid] = np.argsort(np.argsort(row[valid]))
    return out


def _row_corr(rank_a: np.ndarray, rank_b: np.ndarray, min_names: int) -> np.ndarray:
    """Per-row Pearson on rank matrices (== Spearman on raw), NaN-aware."""
    ics = np.full(rank_a.shape[0], np.nan)
    for r in range(rank_a.shape[0]):
        x, y = rank_a[r], rank_b[r]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < min_names:
            continue
        xv = x[m] - x[m].mean()
        yv = y[m] - y[m].mean()
        denom = np.sqrt((xv ** 2).sum() * (yv ** 2).sum())
        if denom > 0:
            ics[r] = (xv * yv).sum() / denom
    return ics


def cross_sectional_ic(signal_panel: pd.DataFrame,
                       fwd_ret: pd.DataFrame,
                       method: str = "spearman",
                       min_names: int = 10) -> pd.Series:
    """Per-date cross-sectional rank IC between signal and forward return.

    Vectorized: rank each row once, Pearson on ranks (= Spearman). Returns a
    Series indexed by date with NaN dates dropped.
    """
    common = signal_panel.index.intersection(fwd_ret.index)
    sig = signal_panel.loc[common].to_numpy(dtype=float)
    ret = fwd_ret.loc[common].to_numpy(dtype=float)
    ic = _row_corr(_row_rank(sig), _row_rank(ret), min_names)
    return pd.Series(ic, index=common).dropna().sort_index()


def ic_summary(ic: pd.Series) -> Dict[str, float]:
    """Headline IC stats: mean, std, IR, t-stat, hit rate, n."""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"n": 0}
    mean = ic.mean()
    std = ic.std(ddof=1) if n > 1 else np.nan
    ir = mean / std if std and std > 0 else np.nan
    tstat = ir * np.sqrt(n) if pd.notna(ir) else np.nan
    return {
        "n": n,
        "mean_ic": round(float(mean), 4),
        "ic_std": round(float(std), 4) if pd.notna(std) else None,
        "ic_ir": round(float(ir), 4) if pd.notna(ir) else None,
        "ic_tstat": round(float(tstat), 4) if pd.notna(tstat) else None,
        "hit_rate": round(float((ic > 0).mean()), 4),
    }


def quantile_long_short(signal_panel: pd.DataFrame,
                        fwd_ret: pd.DataFrame,
                        n_quantiles: int = 5,
                        long_only: bool = False,
                        min_names: int = 10) -> pd.Series:
    """Daily long-short (top vs bottom quantile) return series.

    Each date: rank by signal into n_quantiles, long top quantile,
    short bottom quantile (unless long_only). Equal-weighted within quantile.
    Returns a daily return Series.
    """
    common_dates = signal_panel.index.intersection(fwd_ret.index)
    daily = {}
    for d in common_dates:
        s = signal_panel.loc[d]
        r = fwd_ret.loc[d]
        joined = pd.concat([s, r], axis=1, keys=["s", "r"]).dropna()
        if len(joined) < min_names or joined["s"].std() == 0:
            continue
        try:
            q = pd.qcut(joined["s"].rank(method="first"), n_quantiles,
                        labels=False, duplicates="drop")
        except ValueError:
            continue
        top = joined["r"][q == q.max()].mean()
        if long_only:
            daily[d] = top
        else:
            bot = joined["r"][q == q.min()].mean()
            daily[d] = top - bot
    return pd.Series(daily).sort_index()


def equity_curve_from_returns(daily_ret: pd.Series,
                              initial: float = 1.0) -> pd.Series:
    """Compound a daily return series into an equity curve."""
    return initial * (1.0 + daily_ret.fillna(0.0)).cumprod()
