"""Tests for document ingestion pipeline."""

import pytest
from app.ingestion.chunker import chunk_document, Chunk
from app.ingestion.parser import parse_document


class TestChunker:
    def test_basic_prose_chunking(self):
        text = "Revenue grew strongly. " * 100
        chunks = chunk_document(text)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_table_kept_atomic(self):
        table_text = (
            "Revenue | Q1 2024 | Q1 2023\n"
            "Net Sales | $94.9B | $91.0B\n"
            "EPS | $1.53 | $1.52\n"
        )
        prose = "Some prose before the table.\n\n"
        text = prose + table_text + "\n\nSome prose after."
        chunks = chunk_document(text)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) == 1
        # All table rows should be in one chunk
        assert "Net Sales" in table_chunks[0].text
        assert "EPS" in table_chunks[0].text

    def test_source_label_propagated(self):
        chunks = chunk_document("Hello world " * 20, source_label="Q2 2024 10-Q")
        assert all(c.source_label == "Q2 2024 10-Q" for c in chunks)

    def test_empty_text_returns_empty(self):
        chunks = chunk_document("")
        assert chunks == []

    def test_chunk_indices_sequential(self):
        text = "Word " * 500
        chunks = chunk_document(text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


class TestParser:
    def test_plain_text_passthrough(self):
        raw = b"Hello world"
        assert parse_document(raw, fmt="text") == "Hello world"

    def test_html_strips_scripts(self):
        html = b"<html><head><script>alert(1)</script></head><body><p>Revenue grew.</p></body></html>"
        text = parse_document(html, fmt="html")
        assert "alert" not in text
        assert "Revenue grew" in text

    def test_html_converts_tables(self):
        html = b"""
        <html><body>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Revenue</td><td>$94.9B</td></tr>
        </table>
        </body></html>
        """
        text = parse_document(html, fmt="html")
        assert "Revenue" in text
        assert "$94.9B" in text
        assert "|" in text  # pipe-delimited
