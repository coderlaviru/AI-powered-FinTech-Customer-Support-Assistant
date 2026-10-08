"""Local sentence-transformers embeddings (no API key, no network after the first model download)."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Cosine-ready embeddings: vectors are L2-normalised.

    ``query_prefix`` is prepended to queries only. BGE-style retrieval models are trained with an
    instruction on the query side and lose accuracy without it.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        batch_size: int = 32,
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._factory = model_factory
        self._model = None
        self._warned_truncation = False

    def _load(self):
        if self._model is None:
            if self._factory is not None:
                self._model = self._factory(self._model_name)
            else:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
        return self._model

    def _warn_if_truncated(self, texts: list[str]) -> None:
        if self._warned_truncation:
            return
        model = self._load()
        tokenizer = getattr(model, "tokenizer", None)
        limit = getattr(model, "max_seq_length", None)
        if tokenizer is None or not limit:
            return
        over = sum(1 for text in texts if len(tokenizer.encode(text, add_special_tokens=True)) > limit)
        if over:
            self._warned_truncation = True
            logger.warning(
                "%d chunk(s) exceed the embedding model's %d-token limit and are truncated; "
                "lower CHUNK_MAX_WORDS to avoid losing their tail.",
                over,
                limit,
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, vector)) for vector in vectors]

    def get_text_embedding_batch(self, texts: list[str], **_) -> list[list[float]]:
        self._warn_if_truncated(texts)
        return self._encode(texts)

    def get_query_embedding(self, query: str) -> list[float]:
        return self._encode([self._query_prefix + query])[0]
