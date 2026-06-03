"""Synthetic test: prove the stage-2 engine is correct BEFORE touching live data.

We build a fake universe two ways:
  (A) SIGNAL universe — events genuinely predict next-day returns. The engine
      should report positive mean IC, low Monte-Carlo p-value, positive
      long-short Sharpe.
  (B) NOISE universe — event scores are random, unrelated to returns. The
      engine should report ~0 IC and a high (non-significant) p-value.

If the engine flags (A) as significant and (B) as not, the measurement
machinery is trustworthy. This is the same logic as the FinBERT contamination
check: a tool that can't tell signal from noise can't validate anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import signals as S
import validation as V


def make_universe(n_tickers=30, n_days=350, seed=0, signal_strength=0.0):
    """Generate synthetic OHLCV + an event panel.

    signal_strength=0  -> events are noise (control)
    signal_strength>0  -> event score predicts next-day return with this beta
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    # base random walk returns per ticker
    base = rng.normal(0.0003, 0.02, size=(n_days, n_tickers))

    # sprinkle events: ~8% of (ticker, day) cells get an event
    ev_rows = []
    event_effect = np.zeros((n_days, n_tickers))
    for j, tk in enumerate(tickers):
        for t in range(n_days):
            if rng.random() < 0.08:
                score = float(np.clip(rng.normal(0, 0.5), -1, 1))
                etype = rng.choice(["earnings", "8-K", "insider", "10-Q"])
                ev_rows.append({
                    "date": dates[t], "ticker": tk,
                    "event_type": etype, "score": score,
                })
                # if signal_strength>0, the event nudges the NEXT day's return
                if t + 1 < n_days:
                    event_effect[t + 1, j] += signal_strength * score

    rets = base + event_effect
    prices = {}
    for j, tk in enumerate(tickers):
        close = 100 * np.cumprod(1 + rets[:, j])
        prices[tk] = pd.DataFrame({"close": close}, index=dates)

    events = pd.DataFrame(ev_rows)
    return prices, events


def evaluate(prices, events, label):
    panel = S.build_signal_panel(prices, events, alpha=0.6, typed_decay=True)
    fwd = S.forward_returns(prices)
    ic = S.cross_sectional_ic(panel, fwd, min_names=10)
    summ = S.ic_summary(ic)
    ls = S.quantile_long_short(panel, fwd, n_quantiles=5, min_names=10)
    eq = S.equity_curve_from_returns(ls)
    mc = V.monte_carlo_ic_test(panel, fwd, S.cross_sectional_ic,
                               n_simulations=300, seed=1)
    boot = V.bootstrap_sharpe_ci(ls, n_bootstrap=500)
    wf = V.walk_forward_ic(ic, ls, n_windows=5)

    print(f"\n===== {label} =====")
    print("IC summary      :", summ)
    print("Monte Carlo     :", mc)
    print("Bootstrap Sharpe:", {k: boot[k] for k in
          ("observed_sharpe", "ci_lower", "ci_upper", "prob_positive") if k in boot})
    print("Walk-forward    : consistency =", wf.get("consistency"),
          "| per-window IC =", [w["mean_ic"] for w in wf.get("per_window", [])])
    print("Total return    :", round(float(eq.iloc[-1] / eq.iloc[0] - 1), 4))
    return summ, mc


if __name__ == "__main__":
    # (A) real signal injected
    p_sig, e_sig = make_universe(signal_strength=0.015, seed=7)
    summ_a, mc_a = evaluate(p_sig, e_sig, "A: REAL SIGNAL (should be significant)")

    # (B) pure noise
    p_noise, e_noise = make_universe(signal_strength=0.0, seed=7)
    summ_b, mc_b = evaluate(p_noise, e_noise, "B: NOISE (should NOT be significant)")

    print("\n===== VERDICT =====")
    ok_a = summ_a.get("mean_ic", 0) > 0 and mc_a.get("p_value", 1) < 0.05
    ok_b = abs(summ_b.get("mean_ic", 0)) < 0.03 and mc_b.get("p_value", 0) > 0.05
    print(f"Detects real signal as significant : {ok_a}")
    print(f"Detects noise as NOT significant   : {ok_b}")
    print(f"ENGINE TRUSTWORTHY                 : {ok_a and ok_b}")
