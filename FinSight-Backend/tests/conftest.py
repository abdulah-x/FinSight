"""Shared setup. Nothing here fakes a service: tests either check pure code with real
inputs, or call the real APIs (Groq, Tavily, Yahoo, Binance, ...) and a real Postgres.

    pytest            # pure-code tests only, ~5s, no network
    pytest -m live    # everything live (~15 min: Groq's free tier allows ~8K tokens/minute)

A live run uses ~60-70K gpt-oss-120b tokens: a third of Groq's free 200K tokens/day,
which the running app shares. Run it once per change, not in a loop.
"""

import os
import sys
import time

import psycopg
import pytest
from dotenv import load_dotenv
from langchain_groq import ChatGroq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Real keys from the project .env (needed even to import tools.py and graph.py).
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
# Tests never touch the forward-test table: storage is off unless a test asks for
# the separate finsight_test database (test_db fixture). Set to "" rather than removed:
# load_dotenv() (e.g. on importing collector.py) never overwrites an existing variable.
REAL_DATABASE_URL = os.environ.get("DATABASE_URL") or None
os.environ["DATABASE_URL"] = ""

import cache  # noqa: E402
import db  # noqa: E402

GROQ_PAUSE_SECONDS = 65  # one full analysis uses ~5K of the free 8K tokens/minute; a 45s gap was too tight
_last_full_analysis = [0.0]


@pytest.fixture(autouse=True)
def empty_cache():
    """Each test starts with no cached analyses, so tests can't leak into each other.
    (The calendar download cache is kept: Forex Factory rate-limits repeat downloads.)"""
    cache.clear()


@pytest.fixture
def groq_budget():
    """For tests that run a full analysis: wait until the last one is far enough back
    that Groq's per-minute token budget has refilled."""
    wait = _last_full_analysis[0] + GROQ_PAUSE_SECONDS - time.time()
    if wait > 0:
        time.sleep(wait)
    yield
    _last_full_analysis[0] = time.time()


@pytest.fixture
def test_db(monkeypatch):
    """A real, empty predictions table in its own database (finsight_test), on the same
    Postgres container as the app, so tests never mix with forward-test data."""
    if not REAL_DATABASE_URL:
        pytest.fail("DATABASE_URL is not set in .env (start the postgres container)")
    url = REAL_DATABASE_URL.rsplit("/", 1)[0] + "/finsight_test"
    with psycopg.connect(REAL_DATABASE_URL, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'finsight_test'").fetchone():
            conn.execute("CREATE DATABASE finsight_test")
    monkeypatch.setenv("DATABASE_URL", url)
    db.init_db()
    with psycopg.connect(url) as conn:
        conn.execute("TRUNCATE predictions")
    return url


def rows(url: str, where: str = "TRUE") -> list[tuple]:
    """(asset, timeframe, bias, source, entry_price, exit_price, correct) from the test database."""
    with psycopg.connect(url) as conn:
        return conn.execute(f"SELECT asset, timeframe, bias, source, entry_price, exit_price, correct "
                            f"FROM predictions WHERE {where} ORDER BY id").fetchall()


def unreachable_groq(model: str = "openai/gpt-oss-20b") -> ChatGroq:
    """A real Groq client with a revoked key: every call genuinely fails (401), which is
    how we test what happens when the provider is down."""
    return ChatGroq(model=model, api_key="gsk_invalid_key_for_tests", max_retries=0, timeout=10)
