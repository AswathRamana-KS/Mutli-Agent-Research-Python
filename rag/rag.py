"""
Local RAG Stack
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import chromadb

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 30]

# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------
def _embed_ollama(texts: List[str], model: str, base_url: str) -> List[List[float]]:
    url = f"{base_url}/api/embed"
    # L2 FIX: Batch embeddings in a single HTTP request for massive speedup
    resp = requests.post(url, json={"model": model, "input": texts}, timeout=120)
    resp.raise_for_status()
    return resp.json().get("embeddings", [])

def embed_texts(texts: List[str], model: str, backend: str, ollama_url: str) -> List[List[float]]:
    if backend == "ollama":
        return _embed_ollama(texts, model, ollama_url)
    else:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model)
        return embedder.encode(texts).tolist()

# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, path: str = "./chroma_db", collection_name: str = "research_kb") -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[dict]) -> None:
        if ids:
            self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Tuple[str, str, float]]:
        # M3 FIX: Prevent ChromaDB crash by capping n_results to collection size
        actual_n = min(n_results, self.count())
        if actual_n == 0: return []
        
        res = self._collection.query(query_embeddings=[query_embedding], n_results=actual_n)
        out = []
        if res and res.get("documents") and res["documents"][0]:
            docs  = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
            for d, m, dist in zip(docs, metas, dists):
                out.append((d, m.get("source", "unknown"), dist))
        return out

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name=name)

# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------
class DocumentIngester:
    def __init__(self, store: VectorStore, embed_model: str, embed_backend: str, ollama_url: str) -> None:
        self._store   = store
        self._model   = embed_model
        self._backend = embed_backend
        self._ollama  = ollama_url

    def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_text(text)
        if not chunks: return 0
        embeddings = embed_texts(chunks, self._model, self._backend, self._ollama)
        ids = [f"{hashlib.md5(c.encode()).hexdigest()[:12]}" for c in chunks]
        metas = [{"source": source}] * len(chunks)
        self._store.add(ids, embeddings, chunks, metas)
        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        text = ""
        ext = Path(file_path).suffix.lower()
        try:
            if ext == ".pdf":
                import pypdf
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif ext == ".docx":
                import docx
                doc = docx.Document(file_path)
                text = "\n".join(para.text for para in doc.paragraphs)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            return self.ingest_text(text, source=Path(file_path).name)
        except Exception:
            return 0

# ---------------------------------------------------------------------------
# Fetcher & Retriever
# ---------------------------------------------------------------------------
def _fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return "" # L3 FIX: Return empty string instead of embedding error messages

class RAGRetriever:
    def __init__(self, store: VectorStore, embed_model: str, embed_backend: str, top_k: int = 5, ollama_url: str = "http://localhost:11434") -> None:
        self._store   = store
        self._model   = embed_model
        self._backend = embed_backend
        self._top_k   = top_k
        self._ollama  = ollama_url

    def is_populated(self) -> bool:
        return self._store.count() > 0

    def retrieve(self, query: str) -> str:
        if not self.is_populated(): return ""
        try:
            q_embed = embed_texts([query], self._model, self._backend, self._ollama)[0]
            hits    = self._store.query(q_embed, n_results=self._top_k)
            if not hits: return ""
            parts = []
            for i, (chunk, source, dist) in enumerate(hits, 1):
                parts.append(f"[{i}] (source: {source}, relevance: {1 - dist:.2f})\n{chunk}")
            return "\n\n".join(parts)
        except Exception as exc:
            return f"[RAG error: {exc}]"