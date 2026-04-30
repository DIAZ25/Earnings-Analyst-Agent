# 📊 Earnings Call Analyst Agent

An autonomous multi-agent RAG system that reads earnings call transcripts and SEC filings, then produces a structured analyst briefing — covering **actual results vs. prior guidance**, **executive sentiment scoring**, and **red-flag detection** — with every cited figure verified against source documents before delivery.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup & Installation](#setup--installation)
6. [Running the API](#running-the-api)
7. [Ingesting Documents](#ingesting-documents)
8. [Querying the System](#querying-the-system)
9. [Running with Docker](#running-with-docker)
10. [Running Tests](#running-tests)
11. [Evaluation Harness](#evaluation-harness)
12. [LangFuse Observability](#langfuse-observability)
13. [API Reference](#api-reference)
14. [Configuration Reference](#configuration-reference)
15. [Phase Roadmap](#phase-roadmap)

---

## What It Does

Analysts at hedge funds and banks spend hours manually reading 10-Qs, cross-referencing prior guidance, and scoring management tone. This system does it in seconds with a verifiable paper trail.

Given a natural-language query like _"Did Apple beat revenue guidance?"_ it:

1. Retrieves relevant passages from the filing and transcript vector stores
2. Grades chunk relevance with an LLM filter
3. Computes beat/miss/inline verdicts with **pure Python arithmetic** (no LLM hallucination risk)
4. Scores executive sentiment: confidence index, red flags, tone delta
5. Audits every cited figure against source documents
6. Reflects and retries up to 3× if figures are ungrounded
7. Returns a clean structured JSON + markdown briefing

---

## Architecture

```
POST /query
     │
     ▼
[Supervisor]          — Classifies query type, initialises state (no LLM)
     │
     ▼
[Financials RAG]      — Retrieves chunks from FAISS, drafts briefing (LLM)
     │
     ▼
[Relevance Grader]    — Filters off-topic chunks (LLM)
     │
     ▼
[Comparator]          — Beat/miss/inline arithmetic (NO LLM — pure Python)
     │
     ▼
[Sentiment Agent]     — Confidence index, red flags, tone delta (LLM)
     │
     ▼
[Figure Grader]       — Audits cited figures vs source docs (LLM)
     │
     ├─── All grounded ──────────────────────────► [Final Node]
     │
     └─── Ungrounded ──► [Reflection Agent] ──► loops back (max 3×)
                              (LLM rewrites query)
```

**Key design decision:** The Comparator node is deliberately LLM-free. Beat/miss/inline verdicts are computed with arithmetic from a JSON guidance cache. This prevents the most dangerous failure mode: a hallucinated verdict.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | latest |
| OpenAI API key | GPT-4o recommended |
| (Optional) LangFuse account | for observability |
| (Optional) Docker + Docker Compose | for containerised deployment |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/earnings-analyst-agent.git
cd earnings-analyst-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will also download the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB).

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-...
EDGAR_USER_AGENT=Your Name your@email.com
```

See [Configuration Reference](#configuration-reference) for all options.

---

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Ingesting Documents

### Option A — Auto-download from SEC EDGAR (US tickers)

```bash
# Ingest the most recent 10-Q for Apple
python scripts/ingest_company.py --ticker AAPL --form 10-Q

# Ingest the last 2 annual reports for Microsoft
python scripts/ingest_company.py --ticker MSFT --form 10-K --limit 2
```

### Option B — Ingest a local file (transcript, PDF, HTML)

```bash
# Ingest a local earnings call transcript (plain text)
python scripts/ingest_company.py \
  --ticker AAPL \
  --file path/to/transcript.txt \
  --doc-type transcript

# Ingest a PDF filing
python scripts/ingest_company.py \
  --ticker AAPL \
  --file path/to/10Q.pdf \
  --doc-type filing
```

### Option C — POST to the ingest API

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "doc_type": "filing",
    "text": "... full document text ...",
    "source_label": "Q2 2024 10-Q"
  }'
```

### Setting prior guidance

Before querying beat/miss/inline, load the prior-period guidance:

```bash
curl -X POST http://localhost:8000/guidance/AAPL \
  -H "Content-Type: application/json" \
  -d '[
    {"metric": "Revenue", "value": 90.0, "period": "Q1 2024", "unit": "USD billions"},
    {"metric": "EPS",     "value": 1.52, "period": "Q1 2024", "unit": "USD"}
  ]'
```

Or copy and edit `data/guidance_cache/AAPL_example.json` directly.

---

## Querying the System

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Did Apple beat revenue guidance?", "ticker": "AAPL"}'
```

### Example response

```json
{
  "ticker": "AAPL",
  "query": "Did Apple beat revenue guidance?",
  "briefing": "# Earnings Analyst Briefing — AAPL\n\n...",
  "sentiment": {
    "confidence_index": 0.74,
    "hedging_count": 6,
    "assertive_count": 22,
    "guidance_specificity": "quantified",
    "red_flags": ["headwinds"],
    "tone_delta": 0.09,
    "summary": "Management projected strong Services growth..."
  },
  "guidance_comparisons": [
    {"metric": "Revenue", "prior_guidance": 90.0, "actual": 92.8,
     "beat_miss_inline": "beat", "variance_pct": 3.1},
    {"metric": "EPS", "prior_guidance": 1.52, "actual": 1.53,
     "beat_miss_inline": "inline", "variance_pct": 0.7}
  ],
  "figures_grounded": true,
  "iterations": 1,
  "trace_id": "a1b2c3d4"
}
```

---

## Running with Docker

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api
```

The `data/` directory is mounted as a volume so vector stores and guidance caches persist between container restarts.

---

## Running Tests

```bash
# All tests
pytest

# With verbose output
pytest -v

# Only unit tests (no API)
pytest tests/test_ingestion.py tests/test_agents.py -v

# With coverage
pip install pytest-cov
pytest --cov=app --cov-report=html
```

> **Note:** `tests/test_api.py` mocks all LLM calls so no OpenAI key is needed to run the test suite.

---

## Evaluation Harness

Create a ground-truth dataset at `data/eval/ground_truth.json`:

```json
[
  {
    "query": "Did Apple beat revenue guidance in Q1 2024?",
    "ticker": "AAPL",
    "expected_verdict": "beat",
    "expected_metric": "Revenue"
  }
]
```

Then run:

```bash
python -m app.evaluation.harness \
  --dataset data/eval/ground_truth.json \
  --output data/eval/results.json
```

The harness exits with code `0` if accuracy ≥ 70%, `1` otherwise — CI-friendly.

---

## LangFuse Observability

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`, every agent node call is traced automatically. The `@observe` decorator is a no-op if LangFuse is not configured, so the system works without it.

In the LangFuse dashboard you get:
- Full trace trees per query
- Per-node latency and token counts  
- Faithfulness scores (are cited figures grounded?)
- Evaluation datasets for regression testing

Sign up free at [langfuse.com](https://langfuse.com).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/ingest` | Ingest raw text for a ticker |
| `POST` | `/ingest/upload` | Ingest a PDF/HTML file (multipart) |
| `POST` | `/guidance/{ticker}` | Set prior-period guidance entries |
| `GET`  | `/guidance/{ticker}` | Retrieve stored guidance entries |
| `POST` | `/query` | Run the full multi-agent pipeline |

Full interactive docs at `/docs` when the server is running.

---

## Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `LLM_MODEL` | | `gpt-4o` | Model for all LLM nodes |
| `EDGAR_USER_AGENT` | For EDGAR auto-ingest | — | `Name email` per SEC policy |
| `LANGFUSE_PUBLIC_KEY` | | — | LangFuse tracing (optional) |
| `LANGFUSE_SECRET_KEY` | | — | LangFuse tracing (optional) |
| `LANGFUSE_HOST` | | `https://cloud.langfuse.com` | LangFuse instance URL |
| `COMPANIES_HOUSE_API_KEY` | For UK filings | — | Companies House API |
| `VECTOR_STORE_DIR` | | `data/vector_stores` | FAISS index directory |
| `GUIDANCE_CACHE_DIR` | | `data/guidance_cache` | JSON guidance cache directory |
| `API_HOST` | | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | | `8000` | FastAPI bind port |

---

## Phase Roadmap

| Phase | What Gets Built | Status |
|-------|----------------|--------|
| 1 | Ingestion pipeline + FAISS RAG | ✅ Complete |
| 2 | LangGraph graph + financials agent | ✅ Complete |
| 3 | Comparator + Sentiment nodes | ✅ Complete |
| 4 | Reflection loop + Figure Grader | ✅ Complete |
| 5 | LangFuse tracing + eval harness | ✅ Complete |
| 6 | FastAPI + Docker + SEC EDGAR auto-ingest | ✅ Complete |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Run tests: `pytest`
4. Open a pull request

## Licence

MIT
