"""
Web Scraper
===========
Primary:  Firecrawl (if API key provided) — clean markdown extraction
Fallback: requests + BeautifulSoup — always available, no key needed

Usage
-----
scraper = WebScraper(firecrawl_api_key="fc-...")   # or None for BS4 only
result  = scraper.scrape("https://example.com")
# result.text, result.title, result.url, result.success
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    url:     str
    title:   str = ""
    text:    str = ""
    success: bool = False
    error:   str = ""
    backend: str = ""   # "firecrawl" | "beautifulsoup" | "requests"


# ---------------------------------------------------------------------------
# Firecrawl scraper
# ---------------------------------------------------------------------------

def _scrape_firecrawl(url: str, api_key: str) -> ScrapeResult:
    """Use Firecrawl's /v1/scrape endpoint — returns clean markdown."""
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "url":     url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    }
    resp = requests.post(
        "https://api.firecrawl.dev/v1/scrape",
        json    = payload,
        headers = headers,
        timeout = 30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise ValueError(data.get("error", "Firecrawl scrape failed"))

    page  = data.get("data", {})
    text  = page.get("markdown") or page.get("content") or ""
    title = page.get("metadata", {}).get("title", "")
    return ScrapeResult(url=url, title=title, text=text, success=True, backend="firecrawl")


# ---------------------------------------------------------------------------
# BeautifulSoup fallback
# ---------------------------------------------------------------------------

def _scrape_bs4(url: str) -> ScrapeResult:
    import requests
    try:
        from bs4 import BeautifulSoup
        HAS_BS4 = True
    except ImportError:
        HAS_BS4 = False

    headers = {"User-Agent": "Mozilla/5.0 (Research Agent)"}
    resp    = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    if HAS_BS4:
        soup  = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else ""
        text  = soup.get_text(separator=" ", strip=True)
    else:
        # Minimal tag stripping without BS4
        title = ""
        text  = re.sub(r"<[^>]+>", " ", resp.text)

    # Normalise whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()
    backend = "beautifulsoup" if HAS_BS4 else "requests"
    return ScrapeResult(url=url, title=title, text=text, success=True, backend=backend)


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------

class WebScraper:
    """
    Scrapes a URL and returns clean text.
    Uses Firecrawl if an API key is available, otherwise falls back to BS4.
    """

    def __init__(self, firecrawl_api_key: Optional[str] = None) -> None:
        self._fc_key = firecrawl_api_key

    def scrape(self, url: str) -> ScrapeResult:
        """Scrape a single URL."""
        if self._fc_key:
            try:
                return _scrape_firecrawl(url, self._fc_key)
            except Exception as exc:
                # Fall through to BS4
                pass
        try:
            return _scrape_bs4(url)
        except Exception as exc:
            return ScrapeResult(url=url, success=False, error=str(exc), backend="failed")

    def scrape_many(self, urls: List[str]) -> List[ScrapeResult]:
        return [self.scrape(u) for u in urls if u.strip()]

    @property
    def has_firecrawl(self) -> bool:
        return bool(self._fc_key)
