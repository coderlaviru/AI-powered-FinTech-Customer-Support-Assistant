"""Dense, hybrid (dense + BM25 via reciprocal rank fusion) and reranked retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from .config import RETRIEVAL_MODES, Settings
from .utils import tokenize

logger = logging.getLogger(__name__)

RRF_K = 60


class QueryEmbedder(Protocol):
    def get_query_embedding(self, query: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class RerankerUnavailable(RuntimeError):
    pass


class CrossEncoderReranker:
    """Lazy-loaded sentence-transformers cross-encoder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerUnavailable("sentence-transformers is not installed.") from exc
            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as exc:  # network / model-download failures
                raise RerankerUnavailable(f"Could not load {self._model_name}: {exc}") from exc
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        model = self._load()
        return [float(s) for s in model.predict([(query, p) for p in passages], show_progress_bar=False)]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    dense_similarity: float
    score: float


@dataclass
class RetrievalResult:
    hits: list[Hit]
    top_similarity: float
    mode: str
    notes: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists; higher score is better."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        collection: Collection,
        embedder: QueryEmbedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ) -> None:
        self._collection = collection
        self._embedder = embedder
        self._settings = settings
        self._reranker = reranker
        self._reranker_failed = False

        stored = collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = list(stored["ids"])
        self._texts: dict[str, str] = dict(zip(self._ids, stored["documents"]))
        self._meta: dict[str, dict] = dict(zip(self._ids, stored["metadatas"]))
        if not self._ids:
            raise RuntimeError("The vector index is empty; rebuild it before querying.")
        self._bm25 = BM25Okapi([tokenize(self._texts[i]) for i in self._ids])
        self._embed_query = lru_cache(maxsize=512)(lambda q: tuple(self._embedder.get_query_embedding(q)))

    def chunk_text(self, chunk_id: str) -> str:
        return self._texts[chunk_id]

    # ------------------------------------------------------------------ catalog

    def documents(self) -> list[dict[str, str]]:
        """Distinct source documents (file name, title, code) for catalog-style questions."""
        seen: dict[str, dict[str, str]] = {}
        for chunk_id in self._ids:
            meta = self._meta[chunk_id]
            entry = seen.setdefault(
                meta["file_name"], {"file_name": meta["file_name"], "title": meta["document_title"], "code": ""}
            )
            if meta.get("section_id") == "0":
                for line in self._texts[chunk_id].splitlines():
                    if line.lower().startswith("document code:"):
                        entry["code"] = line.split(":", 1)[1].strip()
        return sorted(seen.values(), key=lambda d: d["file_name"])

    # ------------------------------------------------------------------ search

    def _dense(self, vector: list[float], n: int) -> list[tuple[str, float]]:
        result = self._collection.query(
            query_embeddings=[vector], n_results=min(n, len(self._ids)), include=["distances"]
        )
        return [(cid, 1.0 - dist) for cid, dist in zip(result["ids"][0], result["distances"][0])]

    def _sparse(self, query: str, n: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:n]
        return [self._ids[i] for i in order if scores[i] > 0]

    def _similarities(self, vector: list[float], ids: list[str], known: dict[str, float]) -> dict[str, float]:
        missing = [i for i in ids if i not in known]
        out = dict(known)
        if missing:
            stored = self._collection.get(ids=missing, include=["embeddings"])
            q = np.asarray(vector, dtype=float)
            for cid, emb in zip(stored["ids"], stored["embeddings"]):
                e = np.asarray(emb, dtype=float)
                denom = float(np.linalg.norm(q) * np.linalg.norm(e)) or 1.0
                out[cid] = float(q @ e) / denom
        return out

    def _rerank(self, query: str, ids: list[str], notes: list[str]) -> list[tuple[str, float]] | None:
        if self._reranker is None or self._reranker_failed:
            return None
        try:
            scores = self._reranker.score(query, [self._texts[i] for i in ids])
        except RerankerUnavailable as exc:
            self._reranker_failed = True
            logger.warning("Reranker disabled, falling back to hybrid ranking: %s", exc)
            notes.append(f"reranker unavailable: {exc}")
            return None
        return sorted(zip(ids, scores), key=lambda item: item[1], reverse=True)

    def retrieve(self, query: str, mode: str | None = None, top_k: int | None = None) -> RetrievalResult:
        mode = mode or self._settings.retrieval_mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}.")
        top_k = top_k or self._settings.top_k
        notes: list[str] = []

        vector = list(self._embed_query(query))
        dense = self._dense(vector, self._settings.candidate_k)
        dense_sim = dict(dense)
        top_similarity = dense[0][1] if dense else 0.0

        effective = mode
        if mode == "dense":
            ranked = dense[:top_k]
        else:
            fused = reciprocal_rank_fusion([[cid for cid, _ in dense], self._sparse(query, self._settings.candidate_k)])
            ranked = fused[:top_k]
            if mode == "hybrid_rerank":
                pool = [cid for cid, _ in fused[: max(self._settings.rerank_pool, top_k)]]
                reranked = self._rerank(query, pool, notes)
                if reranked is None:
                    effective = "hybrid"
                else:
                    ranked = reranked[:top_k]

        ids = [cid for cid, _ in ranked]
        similarities = self._similarities(vector, ids, dense_sim)
        hits = [
            Hit(cid, self._texts[cid], self._meta[cid], similarities.get(cid, 0.0), float(score))
            for cid, score in ranked
        ]
        return RetrievalResult(hits=hits, top_similarity=top_similarity, mode=effective, notes=notes)
