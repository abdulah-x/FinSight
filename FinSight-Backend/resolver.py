"""Grades predictions whose time is up: did price move the way the bias said?

Correct means the move beat a quarter of a typical bar (ATR) in the predicted
direction, or, for Neutral, stayed inside that band. The band stops tiny random
wiggles from counting as wins or losses.

Runs every 15 minutes inside the API (main.py), or by hand:
    python resolver.py
"""

import logging
from datetime import timedelta

import pandas as pd
import yfinance as yf

import db

THRESHOLD_ATR = 0.25

log = logging.getLogger("finsight.resolver")


def grade(bias: str, entry: float, exit_: float, atr: float) -> bool:
    move, band = exit_ - entry, THRESHOLD_ATR * atr
    if bias == "Bullish":
        return move > band
    if bias == "Bearish":
        return move < -band
    return abs(move) <= band  # Neutral: price went nowhere meaningful


def exit_prices(symbol: str, times: list) -> dict:
    """Price at each check time: the open of the first hourly bar at or after it.
    One download per asset covers all its due predictions. A time with no bar yet
    (market closed, data delayed) is left out and retried on the next run."""
    start, end = min(times) - timedelta(hours=2), max(times) + timedelta(days=4)
    bars = yf.Ticker(symbol).history(start=start, end=end, interval="1h")
    if bars.empty:
        return {}
    index = bars.index.tz_convert("UTC")
    prices = {}
    for t in times:
        after = bars["Open"][index >= pd.Timestamp(t).tz_convert("UTC")]
        if len(after):
            prices[t] = float(after.iloc[0])
    return prices


def resolve_due() -> int:
    """Grade every due prediction. Returns how many were graded."""
    due = db.due_predictions()
    graded = 0
    for symbol in {row["asset"] for row in due}:
        rows = [r for r in due if r["asset"] == symbol]
        try:
            prices = exit_prices(symbol, [r["resolve_at"] for r in rows])
        except Exception:
            log.exception("could not fetch exit prices for %s; will retry next run", symbol)
            continue
        for r in rows:
            if r["resolve_at"] in prices:
                exit_ = prices[r["resolve_at"]]
                db.record_outcome(r["id"], exit_, grade(r["bias"], r["entry_price"], exit_, r["atr"]))
                graded += 1
    log.info("resolved %d predictions", graded)
    return graded


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("../.env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db.init_db()  # applies the Stage 7 column to an existing table
    print(f"graded {resolve_due()} predictions")
