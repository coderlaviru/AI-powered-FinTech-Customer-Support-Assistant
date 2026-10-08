"""Persistent Chroma index with content-fingerprint based rebuilds."""

from __future__ import annotations

import json
import logging
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection

from .config import Settings
from .ingestion import Chunk, corpus_fingerprint, load_corpus

logger = logging.getLogger(__name__)

COLLECTION_NAME = "finbase_kb"
EMBED_BATCH = 50


class Embedder(Protocol):
    def get_text_embedding_batch(self, texts: list[str], **kwargs) -> list[list[float]]: ...

    def get_query_embedding(self, query: str) -> list[float]: ...


def _client(settings: Settings) -> chromadb.api.ClientAPI:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.index_dir / "chroma"))


def _manifest_path(settings: Settings):
    return settings.index_dir / "manifest.json"


def _fingerprint(settings: Settings) -> str:
    return corpus_fingerprint(
        settings.data_dir.rglob("*.pdf"),
        settings.embedding_model,
        settings.embedding_query_prefix,
        settings.chunk_max_words,
        settings.chunk_overlap_words,
    )


def _is_current(settings: Settings, collection: Collection, fingerprint: str) -> bool:
    path = _manifest_path(settings)
    if not path.exists() or collection.count() == 0:
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("fingerprint") == fingerprint and manifest.get("count") == collection.count()


def _embed_chunks(chunks: list[Chunk], embedder: Embedder) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors.extend(embedder.get_text_embedding_batch([chunk.text for chunk in batch]))
        logger.info("Embedded %d/%d chunks", min(start + EMBED_BATCH, len(chunks)), len(chunks))
    return vectors


def load_or_build_collection(settings: Settings, embedder: Embedder) -> Collection:
    """Return the Chroma collection, (re)building it when the corpus or parameters changed."""
    client = _client(settings)
    fingerprint = _fingerprint(settings)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    if _is_current(settings, collection, fingerprint):
        logger.info("Loaded existing index (%d chunks)", collection.count())
        return collection

    logger.info("Building index from %s", settings.data_dir)
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    _manifest_path(settings).unlink(missing_ok=True)

    chunks = load_corpus(settings.data_dir, settings.chunk_max_words, settings.chunk_overlap_words)
    vectors = _embed_chunks(chunks, embedder)
    for start in range(0, len(chunks), 500):
        window = slice(start, start + 500)
        collection.add(
            ids=[c.chunk_id for c in chunks[window]],
            documents=[c.text for c in chunks[window]],
            metadatas=[c.metadata for c in chunks[window]],
            embeddings=vectors[window],
        )
    _manifest_path(settings).write_text(
        json.dumps({"fingerprint": fingerprint, "count": len(chunks)}), encoding="utf-8"
    )
    logger.info("Index built: %d chunks", len(chunks))
    return collection
