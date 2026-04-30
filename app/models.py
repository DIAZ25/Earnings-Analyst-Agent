from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    filing = "filing"
    transcript = "transcript"


class IngestRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. AAPL")
    doc_type: DocType
    text: str = Field(..., description="Raw document text")
    source_label: str = Field("", description="Human-readable label, e.g. 'Q2 2024 10-Q'")


class IngestResponse(BaseModel):
    ticker: str
    doc_type: DocType
    chunks_indexed: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language question about the company")
    ticker: str = Field(..., description="Ticker symbol")


class BeatMissInline(str, Enum):
    beat = "beat"
    miss = "miss"
    inline = "inline"
    unknown = "unknown"


class GuidanceComparison(BaseModel):
    metric: str
    prior_guidance: Optional[float] = None
    actual: Optional[float] = None
    beat_miss_inline: BeatMissInline
    variance_pct: Optional[float] = None


class SentimentResult(BaseModel):
    confidence_index: float = Field(..., ge=0, le=1)
    hedging_count: int
    assertive_count: int
    guidance_specificity: str  # "quantified" | "qualitative" | "absent"
    red_flags: List[str]
    tone_delta: float          # change vs prior call, NaN if no prior
    summary: str


class QueryResponse(BaseModel):
    ticker: str
    query: str
    briefing: str
    sentiment: Optional[SentimentResult] = None
    guidance_comparisons: List[GuidanceComparison] = []
    figures_grounded: bool
    iterations: int = Field(1, description="How many reflection loops ran")
    trace_id: str


# ── Guidance Cache ────────────────────────────────────────────────────────────

class GuidanceEntry(BaseModel):
    metric: str
    value: float
    period: str          # e.g. "Q3 2024"
    unit: str = ""       # e.g. "USD billions"
    source_label: str = ""
