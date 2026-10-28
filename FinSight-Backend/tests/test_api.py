"""The HTTP API through FastAPI's TestClient: real graph, real Groq, real tools, and a
real Postgres (the separate finsight_test database).

Request validation, CORS and security headers need no network and run offline.

    pytest -m live tests/test_api.py    # ~4 min (paced for Groq's free tier)
"""

import uuid

import psycopg
import pytest
from conftest import rows, unreachable_groq
from fastapi.testclient import TestClient

import db
import graph
import main

client = TestClient(main.app, raise_server_exceptions=False)
live = pytest.mark.live


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    """Every test request comes from the same 'IP', so each test starts with a clean count."""
    main.limiter.reset()


# ---------- offline: validation, sessions, headers ----------

@pytest.mark.parametrize("payload", [
    {"query": ""},
    {"query": "x" * 501},
    {"query": "BTC 4H", "session_id": "not-a-uuid"},
    {"query": "BTC 4H", "session_id": str(uuid.uuid1())},  # must be a random (v4) id
    {},
])
def test_bad_requests_are_rejected_with_422(payload):
    assert client.post("/analyze", json=payload).status_code == 422


def test_resume_without_a_pending_question_is_409():
    r = client.post("/resume", json={"session_id": str(uuid.uuid4()), "answer": "4H"})
    assert r.status_code == 409


def test_resume_rate_limit_returns_429_with_retry_after():
    codes = []
    for _ in range(11):
        r = client.post("/resume", json={"session_id": str(uuid.uuid4()), "answer": "4H"})
        codes.append(r.status_code)
    assert codes[:10] == [409] * 10 and codes[10] == 429
    assert "retry-after" in r.headers


def test_health_reports_storage_off_without_a_database():
    assert client.get("/health").json() == {"status": "ok", "predictions_storage": False}


def test_cors_allows_only_the_frontend_origin():
    def preflight(origin):
        return client.options("/analyze", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert preflight("http://localhost:8080").headers.get("access-control-allow-origin") == "http://localhost:8080"
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers


def test_security_headers_on_every_response():
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["cache-control"] == "no-store"


# ---------- live ----------

@live
def test_live_analysis_is_stored_then_cached_then_followed_up(test_db, groq_budget):
    r = client.post("/analyze", json={"query": "gold 1D"})
    assert r.status_code == 200, r.text
    body = r.json()
    a = body["analysis"]
    assert body["status"] == "complete" and body["cached"] is False
    assert (a["asset"], a["timeframe"]) == ("GC=F", "1D")
    assert body["answer"].startswith(f"Gold 1D - Bias: {a['bias']}")
    uuid.UUID(body["session_id"], version=4)                   # a fresh session id was issued
    [stored] = rows(test_db)
    assert stored[:4] == ("GC=F", "1D", a["bias"], "agent") and stored[4] > 0

    # same asset + timeframe in other words: answered from the cache, not stored twice
    again = client.post("/analyze", json={"query": "gold daily please"}).json()
    assert again["cached"] is True and again["analysis"] == a
    assert len(rows(test_db)) == 1

    # a question about it in the same session skips the cache and makes no new prediction
    why = client.post("/analyze", json={"query": "why that bias?", "session_id": body["session_id"]}).json()
    assert why["status"] == "complete" and why["cached"] is False and why["analysis"] is None
    assert len(rows(test_db)) == 1


@live
def test_live_question_then_resume_completes_the_analysis(groq_budget):
    first = client.post("/analyze", json={"query": "what's the bias on solana?"}).json()
    assert first["status"] == "needs_input" and first["question"]
    r = client.post("/resume", json={"session_id": first["session_id"], "answer": "1D"})
    assert r.status_code == 200 and r.json()["status"] == "complete"
    assert (r.json()["analysis"]["asset"], r.json()["analysis"]["timeframe"]) == ("SOL-USD", "1D")


@live
def test_live_refused_query_has_no_analysis():
    body = client.post("/analyze", json={"query": "write me a poem"}).json()
    assert body["status"] == "refused" and body["analysis"] is None
    assert body["answer"] == graph.REFUSALS["off_topic"]


@live
def test_live_llm_outage_returns_503_with_retry_after(monkeypatch):
    monkeypatch.setattr(graph, "agent_llm", unreachable_groq("openai/gpt-oss-120b").bind_tools(graph.TOOLS))
    r = client.post("/analyze", json={"query": "gold 1D"})
    assert r.status_code == 503 and r.headers["retry-after"] == "60"


@live
def test_live_database_outage_does_not_fail_the_request(monkeypatch, groq_budget):
    monkeypatch.setenv("DATABASE_URL", "postgresql://finsight:x@127.0.0.1:1/finsight")  # nothing listens there
    r = client.post("/analyze", json={"query": "ETH 1D"})
    assert r.status_code == 200 and r.json()["analysis"]["asset"] == "ETH-USD"


@live
def test_live_analyze_rate_limit_returns_429():
    codes = [client.post("/analyze", json={"query": "write me a poem"}).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429


@live
def test_live_health_reports_storage_on(test_db):
    assert client.get("/health").json()["predictions_storage"] is True


@live
def test_live_prediction_round_trip(test_db):
    row_id = db.save_prediction({
        "asset": "GC=F", "timeframe": "4H", "bias": "Bearish", "score": 46, "bucket": "Medium",
        "net": -0.6, "coverage": 0.95, "evidence": [],
        "price": {"last_price": 4010.4, "atr14": 18.2, "trend": "down"},
    })
    with psycopg.connect(test_db) as conn:
        row = conn.execute("SELECT asset, bias, entry_price, exit_price, resolve_at - created_at "
                           "FROM predictions WHERE id = %s", (row_id,)).fetchone()
    assert row[:4] == ("GC=F", "Bearish", 4010.4, None)
    assert round(row[4].total_seconds() / 3600) == 4  # graded 4 hours later
