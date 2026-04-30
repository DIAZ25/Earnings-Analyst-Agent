"""
FAISS vector store scoped to {TICKER}_{doc_type}.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.ingestion.chunker import Chunk
from app.ingestion.embedder import Embedder


class VectorStore:
    """
    Persistent FAISS index + metadata store for one (ticker, doc_type) pair.

    Files written:
        {store_dir}/{ticker}_{doc_type}.faiss   — FAISS index
        {store_dir}/{ticker}_{doc_type}.meta.pkl — chunk metadata list
    """

    def __init__(
        self,
        ticker: str,
        doc_type: str,
        embedder: Embedder,
        store_dir: str = "data/vector_stores",
    ):
        self.ticker = ticker.upper()
        self.doc_type = doc_type
        self.embedder = embedder
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._base = self.store_dir / f"{self.ticker}_{self.doc_type}"
        self._index_path = Path(str(self._base) + ".faiss")
        self._meta_path = Path(str(self._base) + ".meta.pkl")

        self._index = None
        self._meta: List[dict] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and add chunks to the index."""
        import faiss

        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts).astype("float32")
        dim = vectors.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)  # inner-product (cosine after norm)

        # Normalise for cosine similarity
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

        for chunk in chunks:
            self._meta.append({
                "text": chunk.text,
                "is_table": chunk.is_table,
                "source_label": chunk.source_label,
                "chunk_index": chunk.chunk_index,
            })

        self._save()

    def search(self, query: str, k: int = 6) -> List[Tuple[dict, float]]:
        """Return top-k (metadata, score) pairs for a query."""
        import faiss

        if self._index is None or self._index.ntotal == 0:
            return []

        vec = self.embedder.embed_one(query).astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._meta[idx], float(score)))
        return results

    @property
    def total_chunks(self) -> int:
        return len(self._meta)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._meta, f)

    def _load(self):
        import faiss
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        if self._meta_path.exists():
            with open(self._meta_path, "rb") as f:
                self._meta = pickle.load(f)
