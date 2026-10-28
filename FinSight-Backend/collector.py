"""Builds the forward-test sample: runs the confidence engine on a schedule.

Two traders' requests a day can never tell us whether "High" calls beat "Low" ones;
that needs ~50 graded calls per bucket. So this runs the same engine the agent uses
(gather evidence -> score headlines -> confidence.py) for every asset and timeframe,
with no agent loop and no writer: ~700 small-model tokens per call instead of ~4,500.

Each timeframe is predicted once per bar, so samples don't overlap:
    1H every hour, 4H every 4 hours, 1D daily, 1W weekly.
News comes from the free Google News feed, keeping Tavily's monthly credits for traders.

Runs inside the API when COLLECTOR_ENABLED=true (main.py), or by hand:
    python collector.py            # every asset x timeframe once
    python collector.py 1H 4H      # only these timeframes
"""

import logging
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv("../.env")  # before the imports below: tools.py needs the API keys at import time

import db  # noqa: E402
from confidence import CONFIG, asset_class  # noqa: E402
from graph import analyse  # noqa: E402
from tools import _DEFAULT_NEWS_QUERY, _google_news, gather_evidence  # noqa: E402

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "GC=F"]
SOURCES = {  # everything the engine can use for that asset class (news is fetched separately)
    "crypto": ["price_action", "macro", "calendar", "fear_greed", "funding", "social"],
    "gold": ["price_action", "macro", "calendar", "social"],
}
PAUSE_SECONDS = 3  # spread the news-scoring calls under Groq's 8K tokens/minute

log = logging.getLogger("finsight.collector")


def gold_market_open(now: datetime) -> bool:
    """Gold futures trade Sunday ~23:00 UTC to Friday ~21:00 UTC. A prediction made
    while the market is shut would be graded against a stale price."""
    day, hour = now.weekday(), now.hour  # Monday = 0
    if day == 5:                       # Saturday
        return False
    if day == 6:                       # Sunday: opens late evening
        return hour >= 23
    if day == 4:                       # Friday: closes in the evening
        return hour < 21
    return True


def collect_one(symbol: str, timeframe: str) -> int | None:
    results = gather_evidence.invoke({
        "symbol": symbol, "timeframe": timeframe, "sources": SOURCES[asset_class(symbol)],
        "calendar_hours": CONFIG["event_window_hours"][timeframe],
    })
    try:
        time_range = "day" if timeframe in ("1H", "4H") else "week"
        results["news"] = {"provider": "google_news_rss",
                           "articles": _google_news(_DEFAULT_NEWS_QUERY[symbol], time_range)}
    except Exception as e:
        results["news"] = {"error": f"news unavailable: {e}"}

    _, analysis = analyse(results, symbol, timeframe)
    row_id = db.save_prediction(analysis, source="collector")
    log.info("collected %s %s: %s %s/100 (row %s)", symbol, timeframe,
             analysis["bias"], analysis["score"], row_id)
    return row_id


def collect(timeframes: list[str]) -> int:
    """One prediction per asset for each timeframe. Returns how many were stored."""
    stored = 0
    gold_open = gold_market_open(datetime.now(timezone.utc))
    for timeframe in timeframes:
        for symbol in ASSETS:
            if symbol == "GC=F" and not gold_open:
                continue
            try:
                stored += collect_one(symbol, timeframe) is not None
            except Exception:
                log.exception("collector failed for %s %s", symbol, timeframe)
            time.sleep(PAUSE_SECONDS)
    return stored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db.init_db()
    print(f"stored {collect(sys.argv[1:] or ['1H', '4H', '1D', '1W'])} predictions")
