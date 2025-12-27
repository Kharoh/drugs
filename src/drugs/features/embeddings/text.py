"""Text embedding helper utilities (VoyageAI, OpenAI, or sentence-transformers).

These are optional conveniences; install the matching provider packages before use:
- voyage: ``pip install langchain-voyageai``
- openai: ``pip install openai``
- sentence-transformers: ``pip install sentence-transformers``
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Sequence

import numpy as np

Provider = Literal["voyage", "openai", "sentence-transformers"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return L2-normalized vector with numerical stability guard."""
    n = float(np.linalg.norm(x))
    return x / max(n, eps)


def _split_text(text: str, *, max_chars: int = 12_000) -> List[str]:
    """Split text into approximate-length chunks while keeping paragraphs.

    Parameters
    ----------
    text : str
        Input text to chunk.
    max_chars : int, default=12000
        Maximum characters per chunk before spilling into the next chunk.

    Returns
    -------
    list[str]
        List of chunk strings. Preserves paragraph boundaries when possible.
    """

    text = text.strip()
    if not text:
        return [""]

    parts = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for part in parts:
        p = part.strip()
        if not p:
            continue

        if buf and (buf_len + len(p) + 2) > max_chars:
            chunks.append("\n\n".join(buf).strip())
            buf, buf_len = [], 0

        buf.append(p)
        buf_len += len(p) + 2

    if buf:
        chunks.append("\n\n".join(buf).strip())

    return chunks or [""]


def _pool_chunk_vectors(vectors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted-average pooling for chunk-level vectors."""
    if not vectors:
        return np.zeros((1,), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    w = w / max(float(w.sum()), 1e-12)
    X = np.stack(vectors, axis=0).astype(np.float32)
    return (X * w[:, None]).sum(axis=0)


@dataclass(frozen=True)
class TextEmbedConfig:
    """Configuration for text embedding providers and credentials."""
    provider: Provider = "voyage"
    model: Optional[str] = None
    normalize: bool = True
    max_chars_per_chunk: int = 12_000

    # credentials (leave None to read from env)
    voyage_api_key: Optional[str] = None  # VOYAGE_API_KEY
    openai_api_key: Optional[str] = None  # OPENAI_API_KEY


def make_text_embed_fn(cfg: TextEmbedConfig) -> Callable[[str], np.ndarray]:
    """Create a text embedding callable for the configured provider.

    Parameters
    ----------
    cfg : TextEmbedConfig
        Provider selection, model name, normalization flag, and API keys.

    Returns
    -------
    Callable[[str], np.ndarray]
        Function that consumes a text string and returns a 1D float32 embedding.

    Raises
    ------
    ValueError
        If required API keys are missing or provider is unknown.
    """

    if cfg.provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings  # type: ignore[import]

        api_key = cfg.voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Missing VOYAGE_API_KEY (or cfg.voyage_api_key).")

        model = cfg.model or "voyage-3.5"
        embedder = VoyageAIEmbeddings(voyage_api_key=api_key, model=model)

        def embed_fn(text: str) -> np.ndarray:
            chunks = _split_text(text, max_chars=cfg.max_chars_per_chunk)
            vecs: List[np.ndarray] = []
            weights: List[float] = []
            for c in chunks:
                v = np.asarray(embedder.embed_query(c), dtype=np.float32)
                vecs.append(v)
                weights.append(max(len(c), 1))
            out = _pool_chunk_vectors(vecs, weights)
            return _l2_normalize(out) if cfg.normalize else out

        return embed_fn

    if cfg.provider == "openai":
        from openai import OpenAI  # type: ignore[import]

        api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY (or cfg.openai_api_key).")

        model = cfg.model or "text-embedding-3-large"
        client = OpenAI(api_key=api_key)

        def _embed_one(chunk: str) -> np.ndarray:
            resp = client.embeddings.create(model=model, input=[chunk], encoding_format="float")
            return np.asarray(resp.data[0].embedding, dtype=np.float32)

        def embed_fn(text: str) -> np.ndarray:
            chunks = _split_text(text, max_chars=cfg.max_chars_per_chunk)
            vecs = [_embed_one(c) for c in chunks]
            weights = [max(len(c), 1) for c in chunks]
            out = _pool_chunk_vectors(vecs, weights)
            return _l2_normalize(out) if cfg.normalize else out

        return embed_fn

    if cfg.provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model_name = cfg.model or "sentence-transformers/all-mpnet-base-v2"
        st = SentenceTransformer(model_name)

        def embed_fn(text: str) -> np.ndarray:
            chunks = _split_text(text, max_chars=cfg.max_chars_per_chunk)
            X = st.encode(chunks, normalize_embeddings=False)
            vecs = [np.asarray(v, dtype=np.float32) for v in X]
            weights = [max(len(c), 1) for c in chunks]
            out = _pool_chunk_vectors(vecs, weights)
            return _l2_normalize(out) if cfg.normalize else out

        return embed_fn

    raise ValueError(f"Unknown provider: {cfg.provider}")


__all__ = ["TextEmbedConfig", "make_text_embed_fn"]
