"""
Local RAG Stack
===============
vectorstore.py  — ChromaDB wrapper
ingestion.py    — PDF / TXT / DOCX / URL document loader
retriever.py    — Query interface used by workers

All in one file for clarity; split if the codebase grows.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks of ~chunk_size words."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 30]


# ---------------------------------------------------------------------------
# Embedding backend (lazy import)
# ---------------------------------------------------------------------------

def _embed_ollama(texts: List[str], model: str, base_url: str) -> List[List[float]]:
    url = f"{base_url}/api/embeddings"
    embeddings = []
    for text in texts:
        resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"])
    return embeddings


def _embed_st(texts: List[str], model: str) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    return m.encode(texts, show_progress_bar=False).tolist()


def embed_texts(
    texts:      List[str],
    model:      str,
    backend:    str,
    base_url:   str = "http://localhost:11434",
) -> List[List[float]]:
    if backend == "ollama":
        return _embed_ollama(texts, model, base_url)
    return _embed_st(texts, model)


# ---------------------------------------------------------------------------
# ChromaDB Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around ChromaDB for document storage and retrieval."""

    COLLECTION = "research_docs"

    def __init__(self, path: str = "./chroma_db") -> None:
        import chromadb
        os.makedirs(path, exist_ok=True)
        self._client     = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name     = self.COLLECTION,
            metadata = {"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks:     List[str],
        embeddings: List[List[float]],
        source:     str,
    ) -> int:
        if not chunks:
            return 0
        ids = [
            hashlib.md5(f"{source}:{i}:{c[:40]}".encode()).hexdigest()
            for i, c in enumerate(chunks)
        ]
        metas = [{"source": source, "chunk_idx": i} for i in range(len(chunks))]
        # Upsert — safe to call multiple times
        self._collection.upsert(
            ids        = ids,
            documents  = chunks,
            embeddings = embeddings,
            metadatas  = metas,
        )
        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        n_results:       int = 5,
    ) -> List[Tuple[str, str, float]]:
        """Returns list of (chunk_text, source, distance)."""
        results = self._collection.query(
            query_embeddings = [query_embedding],
            n_results        = n_results,
            include          = ["documents", "metadatas", "distances"],
        )
        items = []
        docs  = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []
        for doc, meta, dist in zip(docs, metas, dists):
            items.append((doc, meta.get("source", ""), float(dist)))
        return items

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(self.COLLECTION)


# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------

class DocumentIngester:
    """Loads documents from files or URLs, chunks them, stores in VectorStore."""

    def __init__(
        self,
        store:    VectorStore,
        embed_model:   str,
        embed_backend: str,
        ollama_url:    str = "http://localhost:11434",
    ) -> None:
        self._store   = store
        self._model   = embed_model
        self._backend = embed_backend
        self._ollama  = ollama_url

    def ingest_file(self, filepath: str | Path) -> int:
        path = Path(filepath)
        text = self._read_file(path)
        return self._process(text, source=path.name)

    def ingest_text(self, text: str, source: str = "manual") -> int:
        return self._process(text, source)

    def ingest_url(self, url: str) -> int:
        text = self._fetch_url(url)
        return self._process(text, source=url)

    def _process(self, text: str, source: str) -> int:
        if not text.strip():
            return 0
        chunks = chunk_text(text)
        if not chunks:
            return 0
        embeds = embed_texts(chunks, self._model, self._backend, self._ollama)
        return self._store.add_chunks(chunks, embeds, source)

    @staticmethod
    def _read_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(errors="replace")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                # ✅ FIX: Do not silently fallback to reading a binary PDF as text. 
                # Raise a clear error so the user knows what is missing.
                raise ImportError("The 'pypdf' package is required to read PDF files. Please run: pip install pypdf")
            except Exception as e:
                return f"Error reading PDF: {e}"
        if suffix in (".docx",):
            try:
                import docx
                doc = docx.Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                raise ImportError("The 'python-docx' package is required to read DOCX files. Please run: pip install python-docx")
            except Exception as e:
                return f"Error reading DOCX: {e}"
        # Fallback for unknown files: read as text
        return path.read_text(errors="replace")

    @staticmethod
    def _fetch_url(url: str) -> str:
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "RAG-Ingester/1.0"})
            resp.raise_for_status()
            # Strip HTML tags crudely
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s{2,}", " ", text)
            return text.strip()
        except Exception as exc:
            return f"Failed to fetch {url}: {exc}"


# ---------------------------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Query interface used by worker agents."""

    def __init__(
        self,
        store:         VectorStore,
        embed_model:   str,
        embed_backend: str,
        top_k:         int = 5,
        ollama_url:    str = "http://localhost:11434",
    ) -> None:
        self._store   = store
        self._model   = embed_model
        self._backend = embed_backend
        self._top_k   = top_k
        self._ollama  = ollama_url

    def retrieve(self, query: str) -> str:
        """Returns formatted context string for prompt injection."""
        if self._store.count() == 0:
            return ""
        try:
            q_embed = embed_texts([query], self._model, self._backend, self._ollama)[0]
            hits    = self._store.query(q_embed, n_results=self._top_k)
            if not hits:
                return ""
            parts = []
            for i, (chunk, source, dist) in enumerate(hits, 1):
                parts.append(f"[{i}] (source: {source}, relevance: {1 - dist:.2f})\n{chunk}")
            return "\n\n".join(parts)
        except Exception as exc:
            return f"[RAG error: {exc}]"

    def is_populated(self) -> bool:
        return self._store.count() > 0
