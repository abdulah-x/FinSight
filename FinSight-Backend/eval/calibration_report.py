"""Calibration report: is the bias right, and does the evidence score mean anything?

Reads every graded prediction and answers:
  1. Hit rate per bucket. If the score is meaningful: High > Medium > Low.
  2. The same predictions graded with dumb strategies. The agent only adds value if
     it beats "follow the EMA trend", which is what the price data says on its own.
  3. Hit rate per asset and timeframe.

Every row is graded with the resolver's rule (move beyond 1/4 ATR), so all
strategies are judged identically. Small samples are flagged, not celebrated.

Usage:
    python eval/calibration_report.py              # all predictions
    python eval/calibration_report.py collector    # only one source (agent | collector)
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from resolver import THRESHOLD_ATR, grade  # noqa: E402

MIN_SAMPLE = 30  # below this a hit rate is mostly noise (a 50% coin's 95% range is ~32-68% at n=30)


def _rate(hits: int, n: int) -> str:
    if n == 0:
        return "   -   (n=0)"
    flag = "" if n >= MIN_SAMPLE else "  <- too few to trust"
    return f"{hits / n:6.1%} (n={n}){flag}"


def baselines(rows: list[dict]) -> dict:
    """Hit rate of simple strategies on exactly the same rows and grading rule."""
    trend_bias = {"up": "Bullish", "down": "Bearish"}
    n = len(rows)
    up = sum(r["exit_price"] - r["entry_price"] > THRESHOLD_ATR * r["atr"] for r in rows)
    down = sum(r["exit_price"] - r["entry_price"] < -THRESHOLD_ATR * r["atr"] for r in rows)
    return {
        "follow the EMA trend": sum(
            grade(trend_bias.get(r["trend"], "Neutral"), r["entry_price"], r["exit_price"], r["atr"]) for r in rows),
        "always Bullish": up,
        "always Neutral": n - up - down,
        # random Bullish/Bearish call: right half the time when price moved beyond the band
        "coin flip (bull/bear)": (up + down) / 2,
    }


def summarize(rows: list[dict]) -> dict:
    by_bucket, by_market, by_bias = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r["correct"])
        by_market[(r["asset"], r["timeframe"])].append(r["correct"])
        by_bias[r["bias"]].append(r["correct"])
    hits = sum(r["correct"] for r in rows)

    order = [by_bucket[b] for b in ("Low", "Medium", "High") if len(by_bucket[b]) >= MIN_SAMPLE]
    rates = [sum(v) / len(v) for v in order]
    return {
        "n": len(rows), "hits": hits,
        "buckets": {b: (sum(by_bucket[b]), len(by_bucket[b])) for b in ("High", "Medium", "Low")},
        "bias": {b: (sum(v), len(v)) for b, v in sorted(by_bias.items())},
        "markets": {k: (sum(v), len(v)) for k, v in sorted(by_market.items())},
        "baselines": baselines(rows),
        # None until at least two buckets have enough data to compare
        "score_is_meaningful": (rates == sorted(rates) and rates[0] < rates[-1]) if len(rates) >= 2 else None,
    }


def report(rows: list[dict]) -> str:
    if not rows:
        return "No graded predictions yet. Run the collector and wait for the resolver."
    s = summarize(rows)
    lines = [f"Graded predictions: {s['n']}   overall hit rate: {_rate(s['hits'], s['n'])}", ""]

    lines.append("1. By evidence-score bucket (meaningful if High > Medium > Low):")
    for b, (h, n) in s["buckets"].items():
        lines.append(f"   {b:<7} {_rate(h, n)}")
    verdict = {True: "YES: higher buckets are right more often",
               False: "NO: the score does not rank calls by accuracy yet",
               None: f"not enough data (need {MIN_SAMPLE}+ in at least two buckets)"}
    lines += [f"   Score meaningful? {verdict[s['score_is_meaningful']]}", ""]

    lines.append("2. Against simple strategies on the same predictions:")
    lines.append(f"   {'FinSight':<24} {_rate(s['hits'], s['n'])}")
    for name, h in s["baselines"].items():
        lines.append(f"   {name:<24} {_rate(round(h), s['n'])}")
    trend = s["baselines"]["follow the EMA trend"]
    lines.append(f"   Beats the trend baseline? {'yes' if s['hits'] > trend else 'no'}"
                 f" ({s['hits']} vs {trend} correct)" + ("" if s["n"] >= MIN_SAMPLE else ", but the sample is too small"))
    lines.append("")

    lines.append("3. By call:")
    for b, (h, n) in s["bias"].items():
        lines.append(f"   {b:<8} {_rate(h, n)}")
    lines.append("")
    lines.append("4. By market:")
    for (asset, tf), (h, n) in s["markets"].items():
        lines.append(f"   {asset:<8} {tf:<3} {_rate(h, n)}")
    return "\n".join(lines)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    import db

    rows = db.resolved_predictions()
    if len(sys.argv) > 1:
        rows = [r for r in rows if r["source"] == sys.argv[1]]
    print(report(rows))
