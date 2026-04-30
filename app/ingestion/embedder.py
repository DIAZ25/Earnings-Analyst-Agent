"""
Sentence-transformer embedder wrapper.
"""

from __future__ import annotations

from typing import List
import numpy as np


class Embedder:
    """Thin wrapper around sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) float32 array of embeddings."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_embedder_instance: Embedder | None = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Return a module-level singleton embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder(model_name)
    return _embedder_instance
