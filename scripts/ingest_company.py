#!/usr/bin/env python3
"""
CLI script to auto-download and ingest SEC EDGAR filings.

Usage:
    python scripts/ingest_company.py --ticker AAPL --form 10-Q
    python scripts/ingest_company.py --ticker MSFT --form 10-K --limit 2
    python scripts/ingest_company.py --ticker AAPL --file path/to/transcript.txt --doc-type transcript

Requires:
    EDGAR_USER_AGENT env var (see .env.example)
"""

import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from app.config import get_settings
from app.ingestion.edgar import EdgarClient
from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import get_embedder
from app.ingestion.vector_store import VectorStore

app = typer.Typer(help="Ingest SEC filings or local files into the vector store.")
console = Console()


@app.command()
def ingest(
    ticker: str = typer.Option(..., help="Ticker symbol, e.g. AAPL"),
    form: str = typer.Option("10-Q", help="SEC form type: 10-Q, 10-K, 8-K"),
    limit: int = typer.Option(1, help="Number of most-recent filings to ingest"),
    file: str = typer.Option("", help="Path to a local file (PDF/HTML/TXT) to ingest instead"),
    doc_type: str = typer.Option("filing", help="'filing' or 'transcript'"),
):
    """Download and index SEC filings for a ticker, or ingest a local file."""
    settings = get_settings()
    embedder = get_embedder()
    ticker = ticker.upper()

    if file:
        _ingest_local_file(file, ticker, doc_type, embedder, settings)
        return

    # Auto-download from EDGAR
    console.print(f"[cyan]Connecting to SEC EDGAR for {ticker} ({form})...[/]")

    try:
        client = EdgarClient(user_agent=settings.edgar_user_agent)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    filings = client.list_filings(ticker, form_type=form, limit=limit)
    if not filings:
        console.print(f"[yellow]No {form} filings found for {ticker}[/]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(filings)} filing(s)[/]")

    for filing in filings:
        console.print(f"\n  Fetching: {filing['filing_date']} ({filing['accession_number']})")
        text = client.fetch_filing_text(ticker, form_type=form)
        if not text:
            console.print(f"  [yellow]Could not retrieve filing text — skipping[/]")
            continue

        _index_text(
            text=text,
            ticker=ticker,
            doc_type=doc_type,
            source_label=f"{form} {filing['filing_date']}",
            embedder=embedder,
            settings=settings,
        )


def _ingest_local_file(file_path: str, ticker: str, doc_type: str, embedder, settings):
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(1)

    console.print(f"[cyan]Reading {path.name}...[/]")
    raw = path.read_bytes()

    if path.suffix.lower() == ".pdf":
        text = parse_document(raw, fmt="pdf")
    elif path.suffix.lower() in (".htm", ".html"):
        text = parse_document(raw, fmt="html")
    else:
        text = raw.decode("utf-8", errors="replace")

    _index_text(
        text=text,
        ticker=ticker,
        doc_type=doc_type,
        source_label=path.name,
        embedder=embedder,
        settings=settings,
    )


def _index_text(text, ticker, doc_type, source_label, embedder, settings):
    console.print(f"  [cyan]Chunking...[/]")
    chunks = chunk_document(text, source_label=source_label)
    console.print(f"  Chunks: {len(chunks)}")

    store = VectorStore(
        ticker=ticker,
        doc_type=doc_type,
        embedder=embedder,
        store_dir=settings.vector_store_dir,
    )
    store.add_chunks(chunks)
    console.print(f"  [green]✓ Indexed {len(chunks)} chunks → {ticker}_{doc_type}[/]")


if __name__ == "__main__":
    app()
