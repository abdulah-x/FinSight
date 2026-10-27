"""The HTTP API, driven through FastAPI's TestClient with fake models (no network, no real DB)."""

import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from conftest import REAL_DATABASE_URL
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import db
import graph
import main
from test_graph import FakeGuard, FakeNewsLLM, ScriptedLLM, asks, fake_gather_evidence, gather

client = TestClient(main.app, raise_server_exceptions=False)


@contextmanager
def models(llm, guard=None, news=None):
    """Fake LLMs + fake evidence tool; db.save_prediction replaced by a recorder."""
    save = MagicMock(return_value=1)
    with patch.object(graph, "agent_llm", llm), patch.object(graph, "llm", llm), \
            patch.object(graph, "guardrail_llm", guard or FakeGuard(asset="GC=F", timeframe="1D", new_analysis=True)), \
            patch.object(graph, "news_llm", news or FakeNewsLLM()), \
            patch.dict(graph.TOOLS_BY_NAME, {"gather_evidence": fake_gather_evidence}), \
            patch.object(main.db, "save_prediction", save):
        yield save


def full_run():
    """An agent that gathers evidence, says READY, then writes an explanation."""
    return ScriptedLLM(gather(), AIMessage(content="READY"), AIMessage(content="Summary: price and dollar agree."))


# ---------- /analyze ----------

def test_analyze_returns_a_structured_analysis_and_stores_it():
    with models(full_run()) as save:
        r = client.post("/analyze", json={"query": "gold 1D"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete" and body["cached"] is False
    assert body["analysis"]["bias"] == "Bearish" and body["analysis"]["asset"] == "GC=F"
    assert body["answer"].startswith("Gold 1D - Bias: Bearish")
    uuid.UUID(body["session_id"], version=4)  # a fresh session id was issued
    save.assert_called_once()
    assert save.call_args.args[0]["price"] == {"last_price": 900.0, "atr14": 10.0, "trend": "down"}


def test_repeat_request_is_served_from_cache_without_llm_or_storage():
    with models(full_run()):
        client.post("/analyze", json={"query": "gold 1D"})
    no_llm = ScriptedLLM()  # would crash if called
    with models(no_llm) as save:
        r = client.post("/analyze", json={"query": "gold daily please"})
    body = r.json()
    assert body["cached"] is True and body["analysis"]["bias"] == "Bearish"
    assert no_llm.received == []
    save.assert_not_called()  # counted once, when it was first made


def test_follow_up_question_bypasses_the_cache():
    with models(full_run()):
        first = client.post("/analyze", json={"query": "gold 1D"}).json()
    llm = ScriptedLLM(AIMessage(content="Because price is under both EMAs."))
    with models(llm, guard=FakeGuard(asset="GC=F", timeframe="1D", new_analysis=False)):
        r = client.post("/analyze", json={"query": "why?", "session_id": first["session_id"]})
    assert r.json()["answer"] == "Because price is under both EMAs." and r.json()["cached"] is False


def test_refused_query_has_no_analysis():
    with models(ScriptedLLM(), guard=FakeGuard(category="off_topic")):
        body = client.post("/analyze", json={"query": "write me a poem"}).json()
    assert body["status"] == "refused" and body["analysis"] is None
    assert body["answer"] == graph.REFUSALS["off_topic"]


@pytest.mark.parametrize("payload", [
    {"query": ""},
    {"query": "x" * 501},
    {"query": "BTC 4H", "session_id": "not-a-uuid"},
    {"query": "BTC 4H", "session_id": str(uuid.uuid1())},  # must be a random (v4) id
    {},
])
def test_bad_requests_are_rejected_with_422(payload):
    assert client.post("/analyze", json=payload).status_code == 422


def test_llm_busy_returns_503_with_retry_after():
    class Broken:
        def invoke(self, messages):
            raise TimeoutError()

    with models(Broken()):
        r = client.post("/analyze", json={"query": "gold 1D"})
    assert r.status_code == 503 and r.headers["retry-after"] == "60"


def test_unexpected_error_is_a_generic_500_without_internals():
    with patch.object(main.agent_graph, "invoke", side_effect=RuntimeError("secret internal detail")):
        r = client.post("/analyze", json={"query": "gold 1D"})
    assert r.status_code == 500
    assert "secret" not in r.text and r.json() == {"detail": "Something went wrong. Please try again."}


def test_storage_failure_does_not_fail_the_request():
    with models(full_run()) as save:
        save.side_effect = psycopg.OperationalError("db down")
        r = client.post("/analyze", json={"query": "gold 1D"})
    assert r.status_code == 200 and r.json()["analysis"]["bias"] == "Bearish"


# ---------- needs_input -> /resume ----------

def test_question_then_resume_completes_the_analysis():
    llm = ScriptedLLM(asks(("ask_user", {"question": "Which timeframe: 1H, 4H, 1D or 1W?"})),
                      gather(), AIMessage(content="READY"), AIMessage(content="Summary."))
    guard = FakeGuard(asset="GC=F", timeframe=None, new_analysis=True)
    with models(llm, guard=guard):
        first = client.post("/analyze", json={"query": "gold?"}).json()
        assert first["status"] == "needs_input"
        assert first["question"] == "Which timeframe: 1H, 4H, 1D or 1W?"

        r = client.post("/resume", json={"session_id": first["session_id"], "answer": "1D"})
    assert r.status_code == 200 and r.json()["status"] == "complete"
    assert r.json()["analysis"]["bias"] == "Bearish"


def test_resume_without_a_pending_question_is_409():
    r = client.post("/resume", json={"session_id": str(uuid.uuid4()), "answer": "4H"})
    assert r.status_code == 409


def test_health():
    assert client.get("/health").json() == {"status": "ok", "predictions_storage": False}


# ---------- the real database (pytest -m live) ----------

@pytest.mark.live
@pytest.mark.skipif(not REAL_DATABASE_URL, reason="DATABASE_URL not set")
def test_live_prediction_round_trip(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", REAL_DATABASE_URL)
    db.init_db()
    row_id = db.save_prediction({
        "asset": "GC=F", "timeframe": "4H", "bias": "Bearish", "score": 46, "bucket": "Medium",
        "net": -0.6, "coverage": 0.95, "evidence": [],
        "price": {"last_price": 4010.4, "atr14": 18.2, "trend": "down"},
    })
    with psycopg.connect(REAL_DATABASE_URL) as conn:
        row = conn.execute("SELECT asset, bias, entry_price, exit_price, resolve_at - created_at "
                           "FROM predictions WHERE id = %s", (row_id,)).fetchone()
        conn.execute("DELETE FROM predictions WHERE id = %s", (row_id,))
    assert row[:4] == ("GC=F", "Bearish", 4010.4, None)
    assert round(row[4].total_seconds() / 3600) == 4  # graded 4 hours later
