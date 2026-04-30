"""
Earnings Call Analyst Agent — FastAPI entrypoint.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import (
    IngestRequest, IngestResponse, DocType,
    QueryRequest, QueryResponse,
    GuidanceEntry,
)
from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import get_embedder
from app.ingestion.vector_store import VectorStore
from app.agents.graph import build_graph

settings = get_settings()

app = FastAPI(
    title="Earnings Call Analyst Agent",
    description="Multi-agent RAG system for earnings call analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared singletons ─────────────────────────────────────────────────────────

_embedder = None
_graph = None


def get_embedder_singleton():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def get_graph_singleton():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Ingest (text body) ────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Ingest a document from raw text."""
    embedder = get_embedder_singleton()
    chunks = chunk_document(req.text, source_label=req.source_label)
    store = VectorStore(
        ticker=req.ticker,
        doc_type=req.doc_type.value,
        embedder=embedder,
        store_dir=settings.vector_store_dir,
    )
    store.add_chunks(chunks)
    return IngestResponse(
        ticker=req.ticker,
        doc_type=req.doc_type,
        chunks_indexed=len(chunks),
        message=f"Indexed {len(chunks)} chunks for {req.ticker} ({req.doc_type.value})",
    )


# ── Ingest (file upload) ──────────────────────────────────────────────────────

@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    ticker: str = Form(...),
    doc_type: DocType = Form(...),
    source_label: str = Form(""),
    file: UploadFile = File(...),
):
    """Ingest a PDF or HTML file via multipart upload."""
    raw_bytes = await file.read()
    content_type = file.content_type or ""

    if "pdf" in content_type or file.filename.endswith(".pdf"):
        text = parse_document(raw_bytes, fmt="pdf")
    else:
        text = parse_document(raw_bytes, fmt="html")

    req = IngestRequest(
        ticker=ticker, doc_type=doc_type,
        text=text, source_label=source_label or file.filename,
    )
    return ingest(req)


# ── Guidance cache ────────────────────────────────────────────────────────────

@app.post("/guidance/{ticker}")
def set_guidance(ticker: str, entries: List[GuidanceEntry]):
    """Store prior-period guidance for a ticker."""
    cache_dir = Path(settings.guidance_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker.upper()}.json"
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    existing.extend([e.model_dump() for e in entries])
    path.write_text(json.dumps(existing, indent=2))
    return {"message": f"Saved {len(entries)} guidance entries for {ticker.upper()}"}


@app.get("/guidance/{ticker}")
def get_guidance(ticker: str):
    path = Path(settings.guidance_cache_dir) / f"{ticker.upper()}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run the multi-agent pipeline and return a structured briefing."""
    trace_id = str(uuid.uuid4())[:8]
    graph = get_graph_singleton()
    embedder = get_embedder_singleton()

    initial_state = {
        "query": req.query,
        "ticker": req.ticker.upper(),
        "trace_id": trace_id,
        "embedder": embedder,
        "retrieved_chunks": [],
        "filtered_chunks": [],
        "draft_briefing": "",
        "final_briefing": "",
        "sentiment": None,
        "guidance_comparisons": [],
        "figures_grounded": False,
        "reflection_count": 0,
        "settings": settings,
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        ticker=req.ticker.upper(),
        query=req.query,
        briefing=result["final_briefing"],
        sentiment=result.get("sentiment"),
        guidance_comparisons=result.get("guidance_comparisons", []),
        figures_grounded=result.get("figures_grounded", False),
        iterations=result.get("reflection_count", 0) + 1,
        trace_id=trace_id,
    )
