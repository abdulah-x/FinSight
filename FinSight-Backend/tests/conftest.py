import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Real keys for live tests; a dummy fallback so offline tests run without any .env.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
os.environ.setdefault("TAVILY_API_KEY", "test-key")

import tools  # noqa: E402


@pytest.fixture(autouse=True)
def empty_calendar_cache():
    """Each test starts with no cached calendar, so tests can't leak into each other."""
    tools._calendar_cache.update(fetched_at=None, events=None)
