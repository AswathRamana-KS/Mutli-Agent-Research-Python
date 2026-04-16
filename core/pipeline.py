"""
ResearchPipeline — the single object that the background thread calls.

It:
  1. (Optionally) ingests local documents into ChromaDB
  2. (Optionally) scrapes URLs and ingests them too
  3. Runs the Manager agent, which orchestrates workers
  4. Returns (FinalReport, ExecutionStats) — all events flow via queue
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import List, Optional

from config import Config
from agents.manager import ManagerAgent
from core.models import (
    EventType,
    ExecutionStats,
    FinalReport,
    PipelineEvent,
)
from rag.rag import DocumentIngester, RAGRetriever, VectorStore
from scraper.web_scraper import WebScraper


class ResearchPipeline:

    def __init__(self, config: Config, event_queue: queue.Queue) -> None:
        self._cfg   = config
        self._queue = event_queue

    # ------------------------------------------------------------------
    # Public entry point (called from background thread)
    # ------------------------------------------------------------------

    def run(
        self,
        query:           str,
        local_files:     Optional[List[str]] = None,
        scrape_urls:     Optional[List[str]] = None,
        use_rag:         bool = False,
        use_scraping:    bool = False,
    ) -> tuple[FinalReport, ExecutionStats]:

        retriever: Optional[RAGRetriever] = None

        # ── RAG setup ────────────────────────────────────────────────
        if use_rag or use_scraping:
            store    = VectorStore(path=self._cfg.chroma_path)
            ingester = DocumentIngester(
                store          = store,
                embed_model    = self._cfg.embed_model,
                embed_backend  = self._cfg.embed_backend,
                ollama_url     = self._cfg.ollama_base_url,
            )
            retriever = RAGRetriever(
                store          = store,
                embed_model    = self._cfg.embed_model,
                embed_backend  = self._cfg.embed_backend,
                top_k          = self._cfg.top_k_rag,
                ollama_url     = self._cfg.ollama_base_url,
            )

            # Ingest local files
            if use_rag and local_files:
                for fp in local_files:
                    self._emit(EventType.LOG,
                               message=f"📂 Ingesting file: {Path(fp).name}")
                    n = ingester.ingest_file(fp)
                    self._emit(EventType.LOG,
                               message=f"  ↳ Added {n} chunks from {Path(fp).name}")

            # Scrape and ingest URLs
            if use_scraping and scrape_urls:
                scraper = WebScraper(firecrawl_api_key=self._cfg.firecrawl_api_key)
                for url in scrape_urls:
                    self._emit(EventType.SCRAPE_START, message=f"🌐 Scraping: {url}")
                    result = scraper.scrape(url)
                    if result.success:
                        n = ingester.ingest_text(result.text, source=url)
                        self._emit(EventType.SCRAPE_DONE,
                                   message=f"  ↳ [{result.backend}] {n} chunks from {url}")
                    else:
                        self._emit(EventType.LOG,
                                   message=f"  ↳ ⚠️ Scrape failed: {result.error}")

            if not retriever.is_populated():
                self._emit(EventType.LOG,
                           message="⚠️ RAG enabled but knowledge base is empty — "
                                   "proceeding without retrieval.")
                retriever = None

        # ── Run the manager pipeline ─────────────────────────────────
        manager = ManagerAgent(config=self._cfg, event_queue=self._queue)
        return manager.run(query=query, retriever=retriever)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _emit(self, event_type: EventType, message: str = "") -> None:
        self._queue.put(PipelineEvent(
            event_type = event_type,
            agent      = "Pipeline",
            message    = message,
        ))
