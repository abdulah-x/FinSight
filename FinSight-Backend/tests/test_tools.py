"""Each tool tested on its own.

Offline tests (default) replace the network with fake data, so we can check the
maths exactly and simulate outages. Live tests hit the real APIs once per tool.

    pytest            # offline, ~2s, no internet needed
    pytest -m live    # real APIs (uses 1 Tavily credit)
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from pydantic import ValidationError

import tools
from tools import (
    get_crypto_fear_greed,
    get_economic_calendar,
    get_funding_rate,
    get_macro_snapshot,
    get_price_action,
    get_social_sentiment,
    search_news,
)


class FakeResponse:
    """Stands in for a requests.Response."""

    def __init__(self, json_data=None, status=200, content=b""):
        self._json, self.status_code, self.content = json_data, status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._json


def rising_bars(n, freq="1h"):
    """Close goes 100, 101, 102... with High/Low exactly 1 either side."""
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1, "Low": close - 1, "Close": close}
    ).set_index(pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC"))


# ---------- get_price_action ----------

@patch("tools.yf.Ticker")
def test_price_action_maths_on_known_data(ticker):
    ticker.return_value.history.return_value = rising_bars(100, "1D")
    r = get_price_action.invoke({"symbol": "BTC-USD", "timeframe": "1D"})
    assert r["last_price"] == 199
    assert r["trend"] == "up"                  # price > EMA20 > EMA50 on a steady rise
    assert r["ema50"] < r["ema20"] < 199
    assert r["atr14"] == 2.0                   # every bar is High-Low = 2
    assert r["high_20_bars"] == 200 and r["low_20_bars"] == 179
    assert r["change_pct_20_bars"] == round((199 / 179 - 1) * 100, 2)


@patch("tools.yf.Ticker")
def test_price_action_4h_is_built_from_1h_bars(ticker):
    ticker.return_value.history.return_value = rising_bars(400, "1h")
    r = get_price_action.invoke({"symbol": "GC=F", "timeframe": "4H"})
    ticker.return_value.history.assert_called_with(period="60d", interval="1h")
    assert r["last_price"] == 499 and r["trend"] == "up"
    assert r["atr14"] == 5.0                   # 4 one-hour bars of +1 -> a 4H bar spans 5


@patch("tools.yf.Ticker")
def test_price_action_falling_market_is_down(ticker):
    bars = rising_bars(100, "1D")
    bars[["Open", "High", "Low", "Close"]] = bars[["Open", "High", "Low", "Close"]].iloc[::-1].values
    ticker.return_value.history.return_value = bars
    assert get_price_action.invoke({"symbol": "ETH-USD", "timeframe": "1D"})["trend"] == "down"


@patch("tools.yf.Ticker")
def test_price_action_too_few_bars_is_an_error(ticker):
    ticker.return_value.history.return_value = rising_bars(30, "1D")
    assert "not enough" in get_price_action.invoke({"symbol": "BTC-USD", "timeframe": "1D"})["error"]


@patch("tools.yf.Ticker", side_effect=ConnectionError("yahoo down"))
def test_price_action_outage_is_an_error_not_a_crash(_):
    assert "yahoo down" in get_price_action.invoke({"symbol": "BTC-USD", "timeframe": "1D"})["error"]


def test_price_action_rejects_unlisted_symbol():
    with pytest.raises(ValidationError):
        get_price_action.invoke({"symbol": "DOGE-USD", "timeframe": "1D"})


# ---------- get_macro_snapshot ----------

def fake_macro_closes():
    days = pd.date_range("2026-01-01", periods=10, freq="D")
    data = {
        ("Close", "DX-Y.NYB"): [100 + i for i in range(10)],
        ("Close", "^TNX"): [4.00 + 0.01 * i for i in range(10)],
        ("Close", "^VIX"): [20.0] * 10,
        ("Close", "^GSPC"): [5000.0] * 9 + [float("nan")],  # stock market not open yet today
    }
    return pd.DataFrame(data, index=days)


@patch("tools.yf.download", return_value=fake_macro_closes())
def test_macro_changes_and_units(_):
    r = get_macro_snapshot.invoke({})
    assert r["dollar_index"]["change_1d"] == round((109 / 108 - 1) * 100, 2)
    assert r["us10y_yield"]["change_1d"] == 1.0    # 4.08 -> 4.09 = +1 basis point
    assert r["us10y_yield"]["change_5d"] == 5.0
    assert r["vix"]["change_5d"] == 0.0


@patch("tools.yf.download", return_value=fake_macro_closes())
def test_macro_each_market_keeps_its_own_last_day(_):
    r = get_macro_snapshot.invoke({})
    assert r["dollar_index"]["as_of"] == "2026-01-10"
    assert r["sp500"]["as_of"] == "2026-01-09"     # the NaN day is skipped, not reported as 0


@patch("tools.yf.download", side_effect=ConnectionError("yahoo down"))
def test_macro_outage_is_an_error(_):
    assert "error" in get_macro_snapshot.invoke({})


# ---------- get_economic_calendar ----------

def event(hours_from_now, country="USD", impact="High", title="CPI"):
    when = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return {"title": title, "country": country, "impact": impact, "date": when.isoformat(),
            "forecast": "0.3%", "previous": ""}


@patch("tools.requests.get")
def test_calendar_keeps_only_usd_medium_high_inside_window(get):
    get.return_value = FakeResponse([
        event(2, title="keep: USD high"),
        event(3, impact="Medium", title="keep: USD medium"),
        event(2, country="EUR", title="drop: not USD"),
        event(2, impact="Low", title="drop: low impact"),
        event(-2, title="drop: already happened"),
        event(30, title="drop: outside 24h window"),
    ])
    r = get_economic_calendar.invoke({"hours_ahead": 24})
    assert [e["event"] for e in r["events"]] == ["keep: USD high", "keep: USD medium"]
    assert r["high_impact_count"] == 1
    assert r["events"][0]["previous"] is None       # empty string from the feed -> None


@patch("tools.requests.get")
def test_calendar_reuses_download_for_30_minutes(get):
    get.return_value = FakeResponse([event(2)])
    get_economic_calendar.invoke({"hours_ahead": 24})
    get_economic_calendar.invoke({"hours_ahead": 48})
    assert get.call_count == 1


@patch("tools.requests.get")
def test_calendar_downloads_again_after_cache_expires(get):
    get.return_value = FakeResponse([event(2)])
    get_economic_calendar.invoke({"hours_ahead": 24})
    tools._calendar_cache["fetched_at"] -= timedelta(minutes=31)
    get_economic_calendar.invoke({"hours_ahead": 24})
    assert get.call_count == 2


@patch("tools.requests.get", return_value=FakeResponse(status=429))
def test_calendar_rate_limit_is_an_error(_):
    assert "429" in get_economic_calendar.invoke({"hours_ahead": 24})["error"]


@pytest.mark.parametrize("hours", [0, 169, 500])
def test_calendar_rejects_out_of_range_window(hours):
    with pytest.raises(ValidationError):
        get_economic_calendar.invoke({"hours_ahead": hours})


# ---------- get_crypto_fear_greed ----------

@patch("tools.requests.get")
def test_fear_greed_today_yesterday_average(get):
    values = [20, 30, 40, 50, 60, 70, 80]  # newest first
    get.return_value = FakeResponse({"data": [
        {"value": str(v), "value_classification": "Extreme Fear" if v == 20 else "x"} for v in values
    ]})
    r = get_crypto_fear_greed.invoke({})
    assert r == {"today": 20, "classification": "Extreme Fear", "yesterday": 30, "avg_7d": 50.0}


@patch("tools.requests.get", return_value=FakeResponse({"unexpected": "shape"}))
def test_fear_greed_bad_payload_is_an_error(_):
    assert "error" in get_crypto_fear_greed.invoke({})


# ---------- get_funding_rate ----------

@patch("tools.requests.get")
def test_funding_rate_converts_symbol_and_units(get):
    get.return_value = FakeResponse([{"fundingRate": "0.0001"}] * 8 + [{"fundingRate": "0.0005"}])
    r = get_funding_rate.invoke({"symbol": "BTC-USD"})
    assert get.call_args.kwargs["params"]["symbol"] == "BTCUSDT"
    assert r["latest_pct_8h"] == 0.05                     # 0.0005 -> 0.05%
    assert r["avg_3d_pct_8h"] == round((0.01 * 8 + 0.05) / 9, 4)
    assert r["latest_annualised_pct"] == 54.8             # 0.05% x 3 a day x 365


@patch("tools.requests.get", return_value=FakeResponse(status=451))
def test_funding_rate_geo_block_is_an_error(_):
    assert "451" in get_funding_rate.invoke({"symbol": "ETH-USD"})["error"]


def test_funding_rate_rejects_gold():
    with pytest.raises(ValidationError):
        get_funding_rate.invoke({"symbol": "GC=F"})


# ---------- search_news ----------

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Gold steady before Fed</title><source>Reuters</source>
<pubDate>Tue, 27 Oct 2026 05:00:00 GMT</pubDate><link>https://news.google.com/very/long</link></item>
</channel></rss>"""


@patch("tools.tavily_search")
def test_news_uses_tavily_first(tavily):
    tavily.invoke.return_value = {"results": [{
        "title": "BTC slips", "url": "https://www.coindesk.com/a", "published_date": "today",
        "content": "Bitcoin   fell\n\n  overnight " + "x" * 500,
    }]}
    r = search_news.invoke({"query": "bitcoin", "time_range": "week"})
    assert r["provider"] == "tavily"
    assert tavily.invoke.call_args.args[0] == {"query": "bitcoin", "topic": "news", "time_range": "week"}
    article = r["articles"][0]
    assert article["source"] == "www.coindesk.com"
    assert article["snippet"].startswith("Bitcoin fell overnight") and len(article["snippet"]) == 200


@patch("tools.requests.get", return_value=FakeResponse(content=RSS))
@patch("tools.tavily_search")
def test_news_falls_back_to_google_when_tavily_fails(tavily, get):
    tavily.invoke.side_effect = RuntimeError("credits exhausted")
    r = search_news.invoke({"query": "gold"})
    assert r["provider"] == "google_news_rss"
    assert r["articles"] == [{"title": "Gold steady before Fed", "source": "Reuters",
                              "published": "Tue, 27 Oct 2026 05:00:00 GMT", "snippet": None, "url": None}]
    assert get.call_args.kwargs["params"]["q"] == "gold when:1d"


@patch("tools.requests.get", return_value=FakeResponse(content=RSS))
@patch("tools.tavily_search")
def test_news_falls_back_when_tavily_finds_nothing(tavily, _):
    tavily.invoke.return_value = {"results": []}
    assert search_news.invoke({"query": "gold"})["provider"] == "google_news_rss"


@patch("tools.requests.get", side_effect=ConnectionError("no internet"))
@patch("tools.tavily_search")
def test_news_both_sources_down_is_one_error(tavily, _):
    tavily.invoke.side_effect = RuntimeError("credits exhausted")
    error = search_news.invoke({"query": "gold"})["error"]
    assert "credits exhausted" in error and "no internet" in error


def test_news_rejects_long_query():
    with pytest.raises(ValidationError):
        search_news.invoke({"query": "x" * 101})


# ---------- get_social_sentiment ----------

def post(body, tag=None):
    return {"body": body, "entities": {"sentiment": {"basic": tag} if tag else None}}


@patch("tools.requests.get")
def test_social_tally_and_clean_samples(get):
    get.return_value = FakeResponse({"messages": [
        post("it&#39;s   going up", "Bullish"), post("moon", "Bullish"),
        post("dump incoming", "Bearish"), post("just watching"),
    ]})
    r = get_social_sentiment.invoke({"symbol": "BTC-USD"})
    assert (r["posts"], r["bullish"], r["bearish"], r["untagged"]) == (4, 2, 1, 1)
    assert r["samples"][0] == "it's going up"                # HTML code decoded, spaces collapsed


@patch("tools.requests.get", return_value=FakeResponse({"messages": []}))
def test_social_gold_maps_to_xauusd(get):
    get_social_sentiment.invoke({"symbol": "GC=F"})
    assert get.call_args.args[0].endswith("/XAUUSD.json")


@patch("tools.requests.get", return_value=FakeResponse(status=403))
def test_social_blocked_is_an_error_not_an_empty_list(_):
    assert "403" in get_social_sentiment.invoke({"symbol": "BTC-USD"})["error"]


# ---------- live: each real API, one at a time ----------

@pytest.mark.live
@pytest.mark.parametrize("tool, args, expected_key", [
    (get_price_action, {"symbol": "BTC-USD", "timeframe": "1H"}, "trend"),
    (get_price_action, {"symbol": "GC=F", "timeframe": "4H"}, "trend"),
    (get_macro_snapshot, {}, "dollar_index"),
    (get_economic_calendar, {"hours_ahead": 168}, "events"),
    (get_crypto_fear_greed, {}, "today"),
    (get_funding_rate, {"symbol": "SOL-USD"}, "latest_pct_8h"),
    (search_news, {"query": "bitcoin"}, "articles"),
    (get_social_sentiment, {"symbol": "ETH-USD"}, "posts"),
], ids=["price BTC 1H", "price gold 4H", "macro", "calendar", "fear_greed",
        "funding SOL", "news", "social ETH"])
def test_live(tool, args, expected_key):
    result = tool.invoke(args)
    assert "error" not in result, result["error"]
    assert expected_key in result
