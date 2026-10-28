import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Real keys for live tests; a dummy fallback so offline tests run without any .env.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
# Offline tests must never write to the real database; db tests opt in with this value.
# Set to "" rather than removed: load_dotenv() (e.g. on importing collector.py) never
# overwrites an existing variable, so the real URL can't sneak back in.
REAL_DATABASE_URL = os.environ.get("DATABASE_URL") or None
os.environ["DATABASE_URL"] = ""

import cache  # noqa: E402
import tools  # noqa: E402


@pytest.fixture(autouse=True)
def empty_caches():
    """Each test starts with no cached calendar or analyses, so tests can't leak into each other."""
    tools._calendar_cache.update(fetched_at=None, events=None)
    cache.clear()
