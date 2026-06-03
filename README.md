# EventAlpha — A Look-Ahead-Controlled LLM Event-Signal Research Framework

EventAlpha tests whether SEC filing text can be turned into a usable post-filing equity signal. It uses an LLM to score filing language, converts the scores into cross-sectional factors, and evaluates them through IC analysis, train/test parameter selection, quantile long/short backtests, transaction-cost adjustment, and robustness checks. A FinBERT baseline is included to compare the LLM signal against a standard financial-text sentiment model.

The main emphasis is timing discipline. Many “LLM + backtest” projects look reasonable but accidentally leak future information into the signal. EventAlpha keeps the filing date, signal date, and return window explicit, so the results are evaluated as a real factor test rather than a text-generation demo. The pipeline runs end to end on public SEC EDGAR filings and yfinance price data.


---

## Headline: controlling and *verifying* LLM look-ahead bias

A large language model's weights already encode how stocks performed after past
events. If you ask it to "score this 2024 earnings release," it can quietly use
hindsight — producing a backtest that looks profitable but could never have been
traded. A prompt that says "don't use the future" cannot fix this, because the
contamination lives in the weights, not the instruction.

EventAlpha treats this as the central validity threat and addresses it in four
layers — three preventive, **one that actually measures the leakage**:

1. **Temporal control (structural).** Events are scored on their knowable filing
   date; post-close filings roll to the next session; the backtest enforces
   `event_date <= trade_date` and trades at **T+1**. No future bar can enter a
   signal.
2. **Point-in-time text only.** The model sees just the filing text (the EX-99.1
   press release when present, else the primary document), at `temperature=0`,
   instructed to score on the text alone.
3. **FinBERT contamination check (the verification).** The identical text is
   re-scored by `ProsusAI/finbert`, a fixed pre-trained classifier with **no
   knowledge of any individual stock's future**. We compare the *information
   coefficient* of the LLM scores against FinBERT's on the same text. If the
   LLM's IC were far above FinBERT's, that gap would flag hindsight leakage. In
   this study the two were comparable — evidence the LLM was
   reading the text, not recalling outcomes.
4. **Out-of-sample validation.** Monte Carlo permutation, bootstrap Sharpe CIs,
   and walk-forward consistency.

> The point isn't that the model never cheats — it's that the framework can
> detect whether it did, with a number, instead of asking you to trust a
> prompt.

---

## Honest result: a near-null on public data (and why)

Across S&P 100 earnings filings (Jan 2024 – May 2025), the sentiment factor showed
**no statistically significant cross-sectional predictive power** at a daily
horizon. The strongest configuration (earnings 8-Ks, T+1) reached an IC of only
~0.009 (Monte Carlo p ~ 0.41); cost-adjusted long/short Sharpe was not positive.
The conclusion was stable across **three text-source iterations** (raw 8-K body →
informative-item filter → EX-99.1 press release) and **two evaluation methods**
(cross-sectional IC + event study).

This is a *deliberately reported null*, not a hidden failure. Two reasons it is
the expected outcome on this data:

- **Large-cap market efficiency.** S&P 100 earnings are absorbed within minutes;
  by a T+1 daily entry there is little drift left. (Post-earnings-announcement
  drift is documented mainly in small / low-coverage names — *a hypothesis this
  framework could test directly, not a claim verified here*.)
- **No point-in-time consensus.** A clean *surprise* signal needs analyst
  estimates as they stood before each release (I/B/E/S-grade). Free snapshots
  report *today's* estimates, which for a 2024 filing is itself look-ahead — so
  consensus is deliberately **not** used.

**The deliverable is the framework, not the signal.** It returns honest results
instead of overfit ones, and is built to plug in a stronger data source:

- point-in-time analyst consensus → a true earnings-*surprise* factor;
- intraday bars → capture drift before it is arbitraged away;
- a small/mid-cap universe → where the effect is documented to live;
- real-time news / transcripts → richer, more frequent event text.

Each swaps in without changing the evaluation harness.

---

## Architecture

Two physically decoupled stages — expensive LLM scoring runs once and is cached;
the numerical backtest re-runs in seconds while you tune parameters.

```
STAGE 1  score_events.py     (LLM, run once -> cache/)
  EDGAR 8-K/10-Q  ->  pick informative items  ->  fetch EX-99.1 / primary text
  ->  validity gate  ->  LLM impact score [-1,1] + confidence  ->  events_panel.csv
  (also saves cache/texts/{accession}.txt so FinBERT scores identical text)

finbert_baseline.py          (free, local, CPU)  -> events_panel_finbert.csv

STAGE 2 (numerical, fast, re-runnable)
  factor_eval.py    single-factor framework: train/test split, parameter
                    selection on train, quantile long/short + turnover-adjusted
                    Sharpe + max drawdown on test, MC / bootstrap / walk-forward
  run_backtest.py   cross-sectional IC run + FinBERT contamination check -> run_card
  event_study.py    small-universe diagnostic: do positive-scored events drift up?
```

Core modules: `signals.py` (event decay + cross-sectional engine),
`validation.py` (MC / bootstrap / walk-forward). The factor is **pure sentiment**
(confidence-weighted, time-decayed); no technical signal is blended in.

> On controls: a momentum baseline is intentionally omitted. On large-cap daily
> horizons both sentiment and momentum are near-zero, so comparing them is
> uninformative — you can't claim incremental information when neither carries
> any. The meaningful control is the FinBERT check above.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add your LLM key + SEC User-Agent (one file, once)

# Stage 1 — score events (cached & resumable). Small test first:
python score_events.py --tickers NVDA,AAPL,MSFT,AMZN,META,GOOGL,AMD,AVGO,JPM,TSLA
# then full universe:
python score_events.py

# FinBERT contamination baseline (free, local, ~440MB one-time download)
python finbert_baseline.py

# Stage 2 — evaluate
python factor_eval.py --train-end 2024-12-31 --horizon 5 --item 2.02
python run_backtest.py --alpha 0.0 --horizon 5     # cross-sectional + FinBERT check
python event_study.py --horizons 1,3,5,10 --item 2.02 --market
```

Validate the engine itself anytime (no key, no network):

```bash
python test_synthetic.py          # detects injected signal, rejects pure noise
python score_events.py --selftest
```

Outputs land in `out/` (`factor_run_card.md`, `run_card.md`) and `cache/`.

---

## ⚠ Scope & limitations

- Research prototype, not production: extension is by editing modules, not a
  formal plugin API; no CI; config via CLI flags + constants.
- Static S&P 100 list → mild survivorship bias.
- Short sample (~14 months) → after train/test split, limited statistical power;
  run cards flag this.
- 8-K/10-Q is one event channel; results are specific to it and to large caps.

## License / data

Reuses two functions (`compute_event_signal`, `combine_signals`) and the
validation methodology from the open-source
[Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) project (MIT). Data: SEC
EDGAR (public) and yfinance. No proprietary data is used.
