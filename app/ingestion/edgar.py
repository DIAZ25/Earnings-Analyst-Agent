"""
SEC EDGAR filing downloader.

Fetches 10-Q / 10-K / 8-K filings for US tickers directly from EDGAR.
Requires EDGAR_USER_AGENT env var per SEC fair-use policy.
"""

from __future__ import annotations

import time
from typing import List, Optional

import requests

_BASE = "https://data.sec.gov"
_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"


class EdgarClient:
    def __init__(self, user_agent: str):
        if not user_agent or "contact@example.com" in user_agent:
            raise ValueError(
                "Set EDGAR_USER_AGENT to 'Your Name your@real-email.com' "
                "as required by SEC fair-use policy."
            )
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    # ── Public ────────────────────────────────────────────────────────────────

    def get_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker → zero-padded CIK string."""
        url = f"{_BASE}/submissions/CIK{ticker.upper().zfill(10)}.json"
        # Try ticker lookup first
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = self._get(tickers_url)
        if resp:
            for entry in resp.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        return None

    def list_filings(
        self,
        ticker: str,
        form_type: str = "10-Q",
        limit: int = 4,
    ) -> List[dict]:
        """
        Return a list of recent filing metadata dicts, newest first.
        Each dict has: accession_number, filing_date, form_type, doc_url
        """
        cik = self.get_cik(ticker)
        if not cik:
            raise ValueError(f"CIK not found for ticker {ticker}")

        url = f"{_BASE}/submissions/CIK{cik}.json"
        data = self._get(url)
        if not data:
            return []

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])

        results = []
        for form, acc, date in zip(forms, accessions, dates):
            if form == form_type:
                acc_clean = acc.replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/full-index/"
                    f"{date[:4]}/QTR{_quarter(date)}/{acc_clean}-index.htm"
                )
                results.append({
                    "accession_number": acc,
                    "filing_date": date,
                    "form_type": form,
                    "doc_url": doc_url,
                    "cik": cik,
                })
                if len(results) >= limit:
                    break
        return results

    def fetch_filing_text(self, ticker: str, form_type: str = "10-Q") -> Optional[str]:
        """
        Download the most recent filing of form_type and return plain text.
        """
        filings = self.list_filings(ticker, form_type, limit=1)
        if not filings:
            return None

        acc = filings[0]["accession_number"].replace("-", "")
        cik = filings[0]["cik"]
        # Get index page
        index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count=1&search_text="
        # Direct approach: grab the primary document from submissions API
        sub_url = f"{_BASE}/submissions/CIK{cik}.json"
        sub = self._get(sub_url)
        if not sub:
            return None

        # Find the primary HTML document
        recent = sub.get("filings", {}).get("recent", {})
        primary_docs = recent.get("primaryDocument", [])
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])

        for form, acc_num, primary in zip(forms, accessions, primary_docs):
            if form == form_type:
                acc_clean = acc_num.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary}"
                resp = self.session.get(doc_url, timeout=30)
                if resp.ok:
                    from app.ingestion.parser import parse_document
                    if primary.endswith(".htm") or primary.endswith(".html"):
                        return parse_document(resp.content, fmt="html")
                    return resp.text
                break

        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, url: str) -> Optional[dict]:
        try:
            time.sleep(0.12)  # SEC rate-limit courtesy delay
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None


def _quarter(date_str: str) -> int:
    month = int(date_str[5:7])
    return (month - 1) // 3 + 1
