"""
Table-aware document chunker.

Financial tables are kept as single atomic chunks so figures like
revenue rows are never split mid-cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    chunk_index: int
    is_table: bool = False
    source_label: str = ""
    metadata: dict = field(default_factory=dict)


# A line that looks like a pipe-delimited table row
_TABLE_ROW_RE = re.compile(r".+\|.+")


def chunk_document(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    source_label: str = "",
) -> List[Chunk]:
    """
    Split a document into chunks, keeping table blocks atomic.

    Parameters
    ----------
    text        : Full document text
    chunk_size  : Target token count per prose chunk (approx chars / 4)
    overlap     : Overlap between consecutive prose chunks in chars
    source_label: Human-readable tag stored in chunk metadata
    """
    chunks: List[Chunk] = []
    idx = 0

    # 1. Separate table blocks from prose
    segments = _split_tables(text)

    prose_buffer = ""

    def flush_prose(buf: str):
        nonlocal idx
        if not buf.strip():
            return
        # Split buffer into sliding windows
        window = chunk_size * 4  # approx char count for chunk_size tokens
        step = window - overlap * 4
        start = 0
        while start < len(buf):
            chunk_text = buf[start : start + window].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=idx,
                    is_table=False,
                    source_label=source_label,
                ))
                idx += 1
            start += step

    for seg_type, seg_text in segments:
        if seg_type == "table":
            flush_prose(prose_buffer)
            prose_buffer = ""
            chunks.append(Chunk(
                text=seg_text.strip(),
                chunk_index=idx,
                is_table=True,
                source_label=source_label,
            ))
            idx += 1
        else:
            prose_buffer += seg_text

    flush_prose(prose_buffer)
    return chunks


def _split_tables(text: str):
    """
    Yield (type, text) tuples where type is 'table' or 'prose'.
    A table block is a run of consecutive pipe-delimited lines.
    """
    lines = text.split("\n")
    segments = []
    buf = []
    in_table = False

    for line in lines:
        is_table_line = bool(_TABLE_ROW_RE.match(line))
        if is_table_line:
            if not in_table and buf:
                segments.append(("prose", "\n".join(buf)))
                buf = []
            in_table = True
            buf.append(line)
        else:
            if in_table and buf:
                segments.append(("table", "\n".join(buf)))
                buf = []
            in_table = False
            buf.append(line)

    if buf:
        segments.append(("table" if in_table else "prose", "\n".join(buf)))

    return segments
