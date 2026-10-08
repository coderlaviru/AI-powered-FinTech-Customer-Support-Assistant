from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings  # noqa: E402
from app.index_store import load_or_build_collection  # noqa: E402
from app.rag_engine import RagEngine  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402

DIM = 512


class HashEmbedder:
    """Deterministic bag-of-words embedder: lets the real pipeline run offline."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = np.zeros(DIM)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).tolist()

    def get_text_embedding_batch(self, texts, **_):
        return [self._vector(t) for t in texts]

    def get_query_embedding(self, query):
        self.calls += 1
        return self._vector(query)


class ScriptedLLM:
    """Returns queued responses; records prompts."""

    def __init__(self, responses=None, default="Answer [S1]."):
        self.responses = list(responses or [])
        self.default = default
        self.prompts: list[str] = []

    def complete(self, prompt, **_):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else self.default

        class _Response:
            pass

        response = _Response()
        response.text = text
        return response


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="https://example.invalid/v1",
        embedding_model="hash-embed",
        embedding_query_prefix="",
        data_dir=BACKEND / "data",
        index_dir=tmp_path / "index",
        chunk_max_words=350,
        chunk_overlap_words=40,
        retrieval_mode="hybrid",
        top_k=5,
        candidate_k=20,
        rerank_pool=12,
        reranker_model="none",
        min_similarity=0.05,
        history_turns=6,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def built_index(tmp_path_factory, embedder):
    settings = make_settings(tmp_path_factory.mktemp("kb"))
    collection = load_or_build_collection(settings, embedder)
    return settings, collection


@pytest.fixture()
def retriever(built_index, embedder):
    settings, collection = built_index
    return HybridRetriever(collection, embedder, settings)


@pytest.fixture()
def make_engine(built_index, embedder):
    settings, collection = built_index

    def _make(llm=None, reranker=None, **overrides):
        s = make_settings(settings.index_dir.parent, **{**{"index_dir": settings.index_dir}, **overrides})
        return RagEngine(s, HybridRetriever(collection, embedder, s, reranker=reranker), llm or ScriptedLLM())

    return _make
