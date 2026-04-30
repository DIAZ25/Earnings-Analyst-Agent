"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_text():
    resp = client.post("/ingest", json={
        "ticker": "TEST",
        "doc_type": "filing",
        "text": "Revenue grew to $10 billion. EPS was $1.50. " * 20,
        "source_label": "Test 10-Q",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "TEST"
    assert data["chunks_indexed"] > 0


def test_guidance_set_and_get():
    ticker = "TESTGUID"
    resp = client.post(f"/guidance/{ticker}", json=[
        {"metric": "Revenue", "value": 10.0, "period": "Q1 2024", "unit": "USD billions"}
    ])
    assert resp.status_code == 200

    resp = client.get(f"/guidance/{ticker}")
    assert resp.status_code == 200
    entries = resp.json()
    assert len(entries) == 1
    assert entries[0]["metric"] == "Revenue"


def test_query_returns_structured_response(monkeypatch):
    """
    Smoke test: patch out LLM calls so we can test the pipeline structure
    without needing an OpenAI key.
    """
    import app.agents.financials_rag as fr
    import app.agents.relevance_grader as rg
    import app.agents.sentiment as sa
    import app.agents.reflection as ref

    monkeypatch.setattr(fr, "_draft_briefing", lambda *a, **kw: "Revenue was $10B.")
    monkeypatch.setattr(rg, "_grade_chunks", lambda q, chunks, s: chunks)
    monkeypatch.setattr(sa, "_llm_sentiment_summary", lambda *a, **kw: ("Positive tone.", 0.1))
    monkeypatch.setattr(ref, "_build_revised_query", lambda **kw: kw.get("original_query", ""))

    # First ingest something
    client.post("/ingest", json={
        "ticker": "SMOKETEST",
        "doc_type": "filing",
        "text": "Revenue was $10 billion. EPS $1.50. " * 30,
        "source_label": "smoke test",
    })

    resp = client.post("/query", json={
        "query": "Did revenue beat guidance?",
        "ticker": "SMOKETEST",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "SMOKETEST"
    assert "briefing" in data
    assert "figures_grounded" in data
    assert "trace_id" in data
