"""Stage 7 forward test: resolver, collector and calibration report (no network, no real DB)."""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import collector
import main
import resolver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import calibration_report  # noqa: E402

T0 = datetime(2026, 10, 28, 8, 15, tzinfo=timezone.utc)


# ---------- resolver: grading rule ----------

@pytest.mark.parametrize("bias, exit_, correct", [
    ("Bullish", 103.0, True),    # +3 > +2.5 band
    ("Bullish", 102.0, False),   # +2 is inside the band: not a real move
    ("Bearish", 97.0, True),
    ("Bearish", 103.0, False),
    ("Neutral", 101.0, True),    # stayed inside +-2.5
    ("Neutral", 96.0, False),
])
def test_grade_needs_a_move_beyond_a_quarter_atr(bias, exit_, correct):
    assert resolver.grade(bias, entry=100.0, exit_=exit_, atr=10.0) is correct


# ---------- resolver: exit price lookup ----------

def hourly(start, opens):
    index = pd.date_range(start, periods=len(opens), freq="1h", tz="UTC")
    return pd.DataFrame({"Open": opens}, index=index)


@patch("resolver.yf.Ticker")
def test_exit_price_is_the_open_of_the_first_bar_at_or_after_the_check_time(ticker):
    ticker.return_value.history.return_value = hourly("2026-10-28 07:00", [10.0, 11.0, 12.0])
    prices = resolver.exit_prices("BTC-USD", [T0])  # 08:15 -> first bar at/after is 09:00
    assert prices == {T0: 12.0}


@patch("resolver.yf.Ticker")
def test_check_time_with_no_bar_yet_is_left_for_the_next_run(ticker):
    ticker.return_value.history.return_value = hourly("2026-10-28 06:00", [10.0, 11.0])  # ends 07:00
    assert resolver.exit_prices("GC=F", [T0]) == {}


def test_resolve_due_grades_and_records_each_prediction():
    due = [{"id": 1, "asset": "BTC-USD", "bias": "Bullish", "entry_price": 100.0, "atr": 10.0, "resolve_at": T0},
           {"id": 2, "asset": "GC=F", "bias": "Bearish", "entry_price": 100.0, "atr": 10.0, "resolve_at": T0}]
    record = MagicMock()
    with patch.object(resolver.db, "due_predictions", return_value=due), \
            patch.object(resolver.db, "record_outcome", record), \
            patch.object(resolver, "exit_prices", side_effect=lambda symbol, times: {T0: 110.0}):
        assert resolver.resolve_due() == 2
    record.assert_any_call(1, 110.0, True)    # bullish, price up 10
    record.assert_any_call(2, 110.0, False)   # bearish, price up 10


def test_one_failing_asset_does_not_stop_the_others():
    due = [{"id": 1, "asset": "BTC-USD", "bias": "Bullish", "entry_price": 100.0, "atr": 10.0, "resolve_at": T0},
           {"id": 2, "asset": "GC=F", "bias": "Bullish", "entry_price": 100.0, "atr": 10.0, "resolve_at": T0}]

    def prices(symbol, times):
        if symbol == "GC=F":
            raise ConnectionError("yahoo down")
        return {T0: 110.0}

    record = MagicMock()
    with patch.object(resolver.db, "due_predictions", return_value=due), \
            patch.object(resolver.db, "record_outcome", record), \
            patch.object(resolver, "exit_prices", side_effect=prices):
        assert resolver.resolve_due() == 1
    record.assert_called_once_with(1, 110.0, True)


# ---------- collector ----------

@pytest.mark.parametrize("when, open_", [
    ("2026-10-28 12:00", True),    # Wednesday
    ("2026-10-30 20:59", True),    # Friday before the close
    ("2026-10-30 21:00", False),   # Friday after the close
    ("2026-10-31 12:00", False),   # Saturday
    ("2026-11-01 22:00", False),   # Sunday before the open
    ("2026-11-01 23:00", True),    # Sunday after the open
])
def test_gold_market_hours(when, open_):
    assert collector.gold_market_open(datetime.fromisoformat(when)) is open_


def test_collect_skips_gold_while_its_market_is_shut_and_survives_failures():
    calls = []

    def one(symbol, timeframe):
        calls.append(symbol)
        if symbol == "ETH-USD":
            raise RuntimeError("binance down")
        return 1

    with patch.object(collector, "collect_one", side_effect=one), \
            patch.object(collector, "gold_market_open", return_value=False), \
            patch.object(collector, "PAUSE_SECONDS", 0):
        stored = collector.collect(["1H"])
    assert calls == ["BTC-USD", "ETH-USD", "SOL-USD"]  # no GC=F
    assert stored == 2                                 # ETH failed, the rest were stored


def test_collect_one_stores_the_engine_result_as_a_collector_prediction():
    evidence = {"price_action": {"trend": "down", "last_price": 900.0, "ema20": 950.0, "ema50": 1000.0,
                                 "atr14": 10.0, "change_pct_20_bars": -5.0},
                "calendar": {"events": []}}
    save = MagicMock(return_value=7)
    with patch.object(collector, "gather_evidence", MagicMock(**{"invoke.return_value": evidence})), \
            patch.object(collector, "_google_news", side_effect=ConnectionError("rss down")), \
            patch.object(collector.db, "save_prediction", save):
        assert collector.collect_one("BTC-USD", "4H") == 7
    analysis, = save.call_args.args
    assert save.call_args.kwargs == {"source": "collector"}
    assert analysis["bias"] == "Bearish" and analysis["price"]["last_price"] == 900.0


# ---------- calibration report ----------

def row(bias, bucket, entry, exit_, trend="mixed", atr=10.0, asset="BTC-USD", tf="1H"):
    return {"bias": bias, "bucket": bucket, "entry_price": entry, "exit_price": exit_, "atr": atr,
            "trend": trend, "asset": asset, "timeframe": tf, "source": "collector",
            "correct": resolver.grade(bias, entry, exit_, atr)}


def test_baselines_are_graded_with_the_same_rule():
    rows = [row("Bearish", "Low", 100, 110, trend="up"),     # price up: trend right, us wrong
            row("Bearish", "Low", 100, 90, trend="down"),    # price down: both right
            row("Neutral", "Low", 100, 101, trend="up")]     # flat: trend wrong, us right
    b = calibration_report.baselines(rows)
    assert b["follow the EMA trend"] == 2
    assert b["always Bullish"] == 1
    assert b["always Neutral"] == 1
    assert b["coin flip (bull/bear)"] == 1.0   # 2 rows moved beyond the band, half of them


def test_score_is_meaningful_only_when_higher_buckets_win_more():
    good = [row("Bullish", "High", 100, 110)] * 30 + [row("Bullish", "Low", 100, 90)] * 30
    bad = [row("Bullish", "High", 100, 90)] * 30 + [row("Bullish", "Low", 100, 110)] * 30
    assert calibration_report.summarize(good)["score_is_meaningful"] is True
    assert calibration_report.summarize(bad)["score_is_meaningful"] is False


def test_small_samples_give_no_verdict():
    rows = [row("Bullish", "High", 100, 110)] * 5 + [row("Bullish", "Low", 100, 90)] * 5
    assert calibration_report.summarize(rows)["score_is_meaningful"] is None
    assert "too few to trust" in calibration_report.report(rows)


def test_empty_report():
    assert "No graded predictions yet" in calibration_report.report([])


# ---------- scheduler ----------

@pytest.mark.parametrize("collector_on, jobs", [
    ("true", {"resolver", "collect_1h", "collect_4h", "collect_1d", "collect_1w"}),
    ("false", {"resolver"}),
])
def test_scheduler_jobs(monkeypatch, collector_on, jobs):
    monkeypatch.setenv("COLLECTOR_ENABLED", collector_on)
    with patch.object(main.db, "enabled", return_value=True):
        scheduler = main.start_scheduler()
    try:
        assert {job.id for job in scheduler.get_jobs()} == jobs
    finally:
        scheduler.shutdown(wait=False)


def test_no_scheduler_without_storage():
    with patch.object(main.db, "enabled", return_value=False):
        assert main.start_scheduler() is None
