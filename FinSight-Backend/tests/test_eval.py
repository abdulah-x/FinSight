"""Stage 7 forward test: resolver, collector, calibration report and scheduler.

The grading rule, market hours and report maths are pure code and run offline. The
resolver and collector run live: real Yahoo prices, real evidence, and a real Postgres
(the separate finsight_test database).

    pytest -m live tests/test_eval.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
import yfinance as yf
from conftest import rows

import collector
import db
import main
import resolver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import calibration_report  # noqa: E402

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
    monkeypatch.setenv("DATABASE_URL", "postgresql://storage-configured")  # jobs are only created, never run
    scheduler = main.start_scheduler()
    try:
        assert {job.id for job in scheduler.get_jobs()} == jobs
        # a job that wakes a few seconds late must still run, not be skipped
        assert all(job.misfire_grace_time == 300 and job.coalesce for job in scheduler.get_jobs())
    finally:
        scheduler.shutdown(wait=False)


def test_no_scheduler_without_storage():
    assert main.start_scheduler() is None  # conftest leaves DATABASE_URL empty


# ---------- live: resolver ----------

live = pytest.mark.live


@live
def test_live_exit_price_is_the_open_of_the_first_hourly_bar_after_the_check_time():
    check = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(minute=15, second=0, microsecond=0)
    bars = yf.Ticker("BTC-USD").history(period="2d", interval="1h")
    next_hour = check.replace(minute=0) + timedelta(hours=1)
    expected = float(bars["Open"][bars.index.tz_convert("UTC") == next_hour].iloc[0])
    assert resolver.exit_prices("BTC-USD", [check]) == {check: expected}


@live
def test_live_check_time_with_no_bar_yet_is_left_for_the_next_run():
    later = datetime.now(timezone.utc) + timedelta(hours=2)
    assert resolver.exit_prices("BTC-USD", [later]) == {}


def _prediction(asset, bias, entry, atr):
    return {"asset": asset, "timeframe": "1H", "bias": bias, "score": 50, "bucket": "Medium", "net": 0.5,
            "coverage": 1.0, "evidence": [], "price": {"last_price": entry, "atr14": atr, "trend": "up"}}


@live
def test_live_resolver_grades_due_predictions_and_skips_unpriceable_ones(test_db):
    btc = yf.Ticker("BTC-USD").history(period="1d", interval="1h")["Close"].iloc[-1]
    eth = yf.Ticker("ETH-USD").history(period="1d", interval="1h")["Close"].iloc[-1]
    db.save_prediction(_prediction("BTC-USD", "Bullish", float(btc), float(btc) * 0.01))
    db.save_prediction(_prediction("ETH-USD", "Neutral", float(eth), float(eth) * 0.01))
    db.save_prediction(_prediction("NOPE-USD", "Bearish", 100.0, 1.0))   # Yahoo has no prices for it
    with psycopg.connect(test_db) as conn:
        conn.execute("UPDATE predictions SET resolve_at = now() - interval '3 hours'")  # all due

    assert resolver.resolve_due() == 2
    for asset, _, bias, _, entry, exit_, correct in rows(test_db, "asset <> 'NOPE-USD'"):
        assert exit_ > 0 and correct is resolver.grade(bias, entry, exit_, entry * 0.01)
    [(*_, exit_, correct)] = rows(test_db, "asset = 'NOPE-USD'")
    assert (exit_, correct) == (None, None)    # left for the next run, didn't stop the others


# ---------- live: collector ----------

@live
def test_live_collect_one_stores_the_engine_result_as_a_collector_prediction(test_db):
    row_id = collector.collect_one("BTC-USD", "4H")
    assert row_id is not None
    [(asset, timeframe, bias, source, entry, exit_, _)] = rows(test_db)
    assert (asset, timeframe, source, exit_) == ("BTC-USD", "4H", "collector", None)
    assert bias in ("Bullish", "Bearish", "Neutral") and entry > 0


@live
def test_live_collect_covers_every_asset_and_skips_gold_when_its_market_is_shut(test_db):
    gold_open = collector.gold_market_open(datetime.now(timezone.utc))
    stored = collector.collect(["1D"])
    expected = {"BTC-USD", "ETH-USD", "SOL-USD"} | ({"GC=F"} if gold_open else set())
    assert stored == len(expected)
    assert {r[0] for r in rows(test_db)} == expected
