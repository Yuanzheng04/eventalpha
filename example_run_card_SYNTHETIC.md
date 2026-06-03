# Event-Driven Cross-Sectional Signal — Run Card

> ⚠ THIS IS A SYNTHETIC-DATA EXAMPLE (injected signal) showing the OUTPUT FORMAT ONLY.
> Real EDGAR data will produce lower, more realistic numbers. Not a real result.


_Generated 2026-06-02T00:44:51.935355+00:00 · config `138ee34d1d92`_

## Config

- Universe: S&P 100 (40 names)
- Window: 2024-01-01 → 2025-05-31
- Event source: EDGAR 8-K/10-Q + LLM point-in-time scoring
- alpha (tech weight): 0.6 · typed decay: True
- Signal lag: T+1 (event_date ≤ trade_date enforced)

## Information Coefficient (the headline result)

- n: 349
- mean_ic: 0.0307
- ic_std: 0.1544
- ic_ir: 0.1986
- ic_tstat: 3.7096
- hit_rate: 0.596

## Long-Short Quintile Portfolio

- total_return: 1.1374
- ann_sharpe: 3.7989
- n_days: 349

## Statistical Validation

**Monte Carlo permutation** (signal vs random pairing)
- actual_mean_ic: 0.0307
- p_value: 0.001
- sim_ic_p95: 0.0134
- sim_ic_p05: -0.0141
- n_simulations: 1000

**Bootstrap Sharpe CI** (long-short)
- observed_sharpe: 3.7989
- ci_lower: 2.1498
- ci_upper: 5.4317
- median_sharpe: 3.7409
- prob_positive: 1.0
- n_bootstrap: 1000

**Walk-Forward** (consistency across 5 windows)
- consistency: 1.0
  - W1 2024-01-01→2024-04-04: IC=0.0278 LS_Sharpe=2.9385
  - W2 2024-04-05→2024-07-10: IC=0.0397 LS_Sharpe=5.203
  - W3 2024-07-11→2024-10-15: IC=0.0458 LS_Sharpe=3.5866
  - W4 2024-10-16→2025-01-20: IC=0.0009 LS_Sharpe=2.1782
  - W5 2025-01-21→2025-05-01: IC=0.0386 LS_Sharpe=5.0723

## Look-Ahead Contamination Check (LLM vs FinBERT)

- LLM mean IC: 0.0307
- FinBERT mean IC: 0.0284
- IC gap (LLM − FinBERT): 0.0023
- interpretation: LLM IC comparable to FinBERT — leakage limited.