"""FinBERT baseline — the look-ahead contamination check.

Re-scores the IDENTICAL filing texts the LLM saw (cache/texts/*.txt) with
ProsusAI/finbert, a 110M-param open model. Produces an event panel on the same
schema so stage 2 can compute FinBERT's IC and compare it to the LLM's IC.

Why this matters: FinBERT is a fixed, pre-trained classifier with NO knowledge
of any individual stock's future. If the LLM's score->forward-return IC is far
ABOVE FinBERT's on identical text, that gap is a red flag for hindsight leakage
in the LLM scores. If they're comparable, the LLM signal plausibly comes from
reading the text, not from memorized outcomes.

COST: zero API cost. Runs locally on CPU. One-time weight download ~440MB.
  pip install transformers torch
First run downloads ProsusAI/finbert from Hugging Face; afterwards fully offline.

Output: cache/events_panel_finbert.csv  (date,ticker,event_type,score,accession,summary)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
TEXTS = CACHE / "texts"


def load_finbert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tok, model, torch


def score_text_finbert(text: str, tok, model, torch) -> float:
    """FinBERT score in [-1, 1] = P(positive) - P(negative)."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
    return float(probs[0] - probs[1])


def main():
    llm_panel = pd.read_csv(CACHE / "events_panel.csv")
    tok, model, torch = load_finbert()
    out_path = CACHE / "events_panel_finbert.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "event_type", "score", "accession", "summary"])
        for _, row in llm_panel.iterrows():
            accn = str(row["accession"])
            tp = TEXTS / f"{accn}.txt"
            if not tp.exists():
                continue
            score = score_text_finbert(tp.read_text(), tok, model, torch)
            w.writerow([row["date"], row["ticker"], row["event_type"],
                        round(score, 4), accn, "finbert"])
    print(f"FinBERT panel -> {out_path}")


if __name__ == "__main__":
    main()
