"""Analysis cache keyed by (asset, timeframe).

A repeat request inside the TTL gets the stored answer: zero tokens, and the same
answer every time instead of slightly different wording. Shorter timeframes go
stale faster, so they expire sooner. In-memory: resets when the server restarts.
"""

import threading
import time

TTL_SECONDS = {"1H": 10 * 60, "4H": 30 * 60, "1D": 2 * 3600, "1W": 12 * 3600}

_store: dict[tuple[str, str], tuple[float, dict]] = {}
_lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool


def get(asset: str, timeframe: str) -> dict | None:
    with _lock:
        entry = _store.get((asset, timeframe))
    if entry is None:
        return None
    stored_at, value = entry
    if time.time() - stored_at > TTL_SECONDS[timeframe]:
        return None
    return value


def put(asset: str, timeframe: str, value: dict) -> None:
    with _lock:
        _store[(asset, timeframe)] = (time.time(), value)


def clear() -> None:
    with _lock:
        _store.clear()
