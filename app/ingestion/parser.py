"""
Document parsing: PDF bytes → text, HTML bytes → text.
"""

from __future__ import annotations

import io
from typing import Literal


def parse_document(raw: bytes, fmt: Literal["pdf", "html", "text"] = "text") -> str:
    """
    Convert raw document bytes to a plain-text string.

    Parameters
    ----------
    raw : bytes
        The raw file content.
    fmt : str
        One of 'pdf', 'html', or 'text'.

    Returns
    -------
    str
        Extracted plain text, preserving table structure where possible.
    """
    if fmt == "pdf":
        return _parse_pdf(raw)
    elif fmt == "html":
        return _parse_html(raw)
    return raw.decode("utf-8", errors="replace")


def _parse_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _parse_html(raw: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    soup = BeautifulSoup(raw, "lxml")

    # Remove scripts / styles
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    # Convert tables to pipe-delimited text so structure survives chunking
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows))

    return soup.get_text(separator="\n")
