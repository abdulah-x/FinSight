"""Each tool tested on its own, against the real APIs.

Argument validation (the allowlist) is pure code and runs offline. Everything else
calls the real source and checks the result is present, fresh and internally consistent.

    pytest -m live tests/test_tools.py    # uses ~2 Tavily credits
"""

import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pandas as pd
import pytest
from langchain_tavily import TavilySearch
from pydantic import ValidationError

import tools
from tools import (
    _google_news,
    gather_evidence,
    get_crypto_fear_greed,
    get_economic_calendar,
    get_funding_rate,
    get_macro_snapshot,
    get_price_action,
    get_social_sentiment,
    search_news,
)

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "GC=F"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
# newest bar older than this = stale data (gold allows for its weekend close)
MAX_AGE_HOURS = {"1H": 3, "4H": 9, "1D": 50, "1W": 24 * 9}
GOLD_EXTRA_HOURS = 72

live = pytest.mark.live


# ---------- the allowlist: rejected before any network call ----------

def test_price_action_rejects_unlisted_symbol():
    with pytest.raises(ValidationError):
        get_price_action.invoke({"symbol": "DOGE-USD", "timeframe": "1D"})


@pytest.mark.parametrize("hours", [0, 169, 500])
def test_calendar_rejects_out_of_range_window(hours):
    with pytest.raises(ValidationError):
        get_economic_calendar.invoke({"hours_ahead": hours})


def test_funding_rate_rejects_gold():
    with pytest.raises(ValidationError):
        get_funding_rate.invoke({"symbol": "GC=F"})


def test_news_rejects_long_query():
    with pytest.raises(ValidationError):
        search_news.invoke({"query": "x" * 101})


@pytest.mark.parametrize("bad", [
    {"sources": []},                          # must ask for at least one
    {"sources": ["horoscope"]},               # not an allowed source
    {"sources": ["news"], "calendar_hours": 500},
    {"sources": ["news"], "news_query": "x" * 101},
])
def test_gather_rejects_bad_arguments(bad):
    with pytest.raises(ValidationError):
        gather_evidence.invoke({"symbol": "BTC-USD", "timeframe": "1H", **bad})


def test_gather_refuses_crypto_only_sources_for_gold_without_fetching():
    r = gather_evidence.invoke({"symbol": "GC=F", "timeframe": "1D", "sources": ["funding", "fear_greed"]})
    assert "crypto only" in r["funding"]["error"] and "crypto only" in r["fear_greed"]["error"]


# ---------- get_price_action (Yahoo) ----------

@live
@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D", "1W"])
@pytest.mark.parametrize("symbol", ASSETS)
def test_live_price_action_is_fresh_and_consistent(symbol, timeframe):
    r = get_price_action.invoke({"symbol": symbol, "timeframe": timeframe})
    assert "error" not in r, r["error"]
    assert (r["symbol"], r["timeframe"]) == (symbol, timeframe)
    assert r["atr14"] > 0
    assert r["low_20_bars"] <= r["last_price"] <= r["high_20_bars"]
    # the trend label follows from the EMAs it reports
    expected = ("up" if r["last_price"] > r["ema20"] > r["ema50"]
                else "down" if r["last_price"] < r["ema20"] < r["ema50"] else "mixed")
    assert r["trend"] == expected

    as_of = pd.Timestamp(r["as_of"])
    age = (datetime.now(timezone.utc) - as_of.tz_convert("UTC")).total_seconds() / 3600
    assert age <= MAX_AGE_HOURS[timeframe] + (GOLD_EXTRA_HOURS if symbol == "GC=F" else 0), f"{age:.1f}h old"
    if timeframe == "4H":
        assert as_of.hour % 4 == 0  # built from 1h bars into 4-hour blocks


# ---------- get_macro_snapshot (Yahoo) ----------

PLAUSIBLE = {"dollar_index": (70, 150), "us10y_yield": (0.5, 10), "vix": (5, 100), "sp500": (1000, 20000)}


@live
def test_live_macro_values_units_and_dates():
    r = get_macro_snapshot.invoke({})
    assert "error" not in r, r.get("error")
    for name, (low, high) in PLAUSIBLE.items():
        item = r[name]
        assert "error" not in item, (name, item)
        assert low <= item["last"] <= high, (name, item)
        assert (datetime.now(timezone.utc).date() - datetime.fromisoformat(item["as_of"]).date()).days <= 5
    assert abs(r["us10y_yield"]["change_1d"]) < 50   # basis points: a day's move is rarely >30bp
    assert abs(r["dollar_index"]["change_1d"]) < 3   # percent


# ---------- get_economic_calendar (Forex Factory) ----------

@live
def test_live_calendar_window_and_impact():
    week = get_economic_calendar.invoke({"hours_ahead": 168})
    day = get_economic_calendar.invoke({"hours_ahead": 24})
    assert "error" not in week, week["error"]
    for e in week["events"]:
        assert e["impact"] in ("High", "Medium") and 0 <= e["hours_away"] <= 168
    assert week["high_impact_count"] == sum(e["impact"] == "High" for e in week["events"])
    assert all(e["hours_away"] <= 24 for e in day["events"])
    assert {e["event"] for e in day["events"]} <= {e["event"] for e in week["events"]}


@live
def test_live_calendar_download_is_reused_for_30_minutes():
    get_economic_calendar.invoke({"hours_ahead": 24})
    first = tools._calendar_cache["fetched_at"]
    get_economic_calendar.invoke({"hours_ahead": 48})
    assert tools._calendar_cache["fetched_at"] == first      # no second download

    tools._calendar_cache["fetched_at"] = first - timedelta(minutes=31)
    r = get_economic_calendar.invoke({"hours_ahead": 24})
    assert "error" not in r, r["error"]
    assert tools._calendar_cache["fetched_at"] > first        # expired: downloaded again


# ---------- get_crypto_fear_greed (alternative.me) ----------

@live
def test_live_fear_greed():
    r = get_crypto_fear_greed.invoke({})
    assert "error" not in r, r["error"]
    assert 0 <= r["today"] <= 100 and 0 <= r["yesterday"] <= 100 and 0 <= r["avg_7d"] <= 100
    assert r["classification"] in {"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}


# ---------- get_funding_rate (Binance) ----------

@live
@pytest.mark.parametrize("symbol", CRYPTO)
def test_live_funding_rate_units(symbol):
    r = get_funding_rate.invoke({"symbol": symbol})
    assert "error" not in r, r["error"]
    assert abs(r["latest_pct_8h"]) < 1                        # in %, so 0.01 is normal
    assert abs(r["latest_annualised_pct"] - r["latest_pct_8h"] * 3 * 365) < 0.2


# ---------- search_news (Tavily, Google News backup) ----------

@live
def test_live_news_from_tavily_is_compact():
    r = search_news.invoke({"query": "bitcoin", "time_range": "day"})
    assert r.get("provider") == "tavily", r
    assert 1 <= len(r["articles"]) <= 5
    for a in r["articles"]:
        assert a["title"] and a["source"] == urlparse(a["url"]).netloc
        assert a["snippet"] is None or (len(a["snippet"]) <= 200 and "  " not in a["snippet"] and "\n" not in a["snippet"])


@live
def test_live_news_falls_back_to_google_when_tavily_rejects_the_key(monkeypatch):
    monkeypatch.setattr(tools, "tavily_search", TavilySearch(max_results=5, tavily_api_key="tvly-invalid-key"))
    r = search_news.invoke({"query": "gold price", "time_range": "week"})
    assert r.get("provider") == "google_news_rss", r
    assert r["articles"] and all(a["title"] and a["url"] is None and a["snippet"] is None for a in r["articles"])


@live
def test_live_google_news_feed():
    articles = _google_news("gold price", "day")
    assert 1 <= len(articles) <= 5 and all(a["title"] and a["source"] for a in articles)


# ---------- get_social_sentiment (StockTwits) ----------

@live
@pytest.mark.parametrize("symbol", ASSETS)
def test_live_social_tally_adds_up(symbol):
    r = get_social_sentiment.invoke({"symbol": symbol})   # GC=F is looked up as XAUUSD
    assert "error" not in r, r["error"]
    assert r["symbol"] == symbol and r["posts"] > 0
    assert r["bullish"] + r["bearish"] + r["untagged"] == r["posts"]
    assert len(r["samples"]) <= 5
    for s in filter(None, r["samples"]):
        assert len(s) <= 150 and "&#39;" not in s and "&amp;" not in s and "\n" not in s


# ---------- gather_evidence ----------

@live
def test_live_gather_fetches_exactly_the_requested_sources():
    r = gather_evidence.invoke({"symbol": "BTC-USD", "timeframe": "4H",
                                "sources": ["price_action", "funding", "fear_greed", "funding"]})
    assert list(r) == ["price_action", "funding", "fear_greed"]   # duplicate fetched once
    assert all("error" not in v for v in r.values()), r
    assert r["price_action"]["timeframe"] == "4H" and r["funding"]["symbol"] == "BTC-USD"


@live
def test_live_gather_gold_runs_the_rest_when_crypto_sources_are_refused():
    r = gather_evidence.invoke({"symbol": "GC=F", "timeframe": "1D", "sources": ["funding", "macro"]})
    assert "crypto only" in r["funding"]["error"]
    assert "dollar_index" in r["macro"]


@live
def test_live_gather_slow_sources_time_out_without_blocking(monkeypatch):
    monkeypatch.setattr(tools, "SOURCE_TIMEOUT_SECONDS", 0.001)  # no real request is this fast
    start = time.time()
    r = gather_evidence.invoke({"symbol": "BTC-USD", "timeframe": "1D", "sources": ["macro", "fear_greed"]})
    assert "timed out" in r["macro"]["error"] and "timed out" in r["fear_greed"]["error"]
    assert time.time() - start < 1
