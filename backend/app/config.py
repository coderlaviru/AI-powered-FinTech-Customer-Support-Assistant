"""Central, validated configuration. Every tunable is read from the environment once."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str
    embedding_model: str
    embedding_query_prefix: str
    data_dir: Path
    index_dir: Path
    chunk_max_words: int
    chunk_overlap_words: int
    retrieval_mode: str
    top_k: int
    candidate_k: int
    rerank_pool: int
    reranker_model: str
    min_similarity: float
    history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("RETRIEVAL_MODE", "hybrid_rerank").strip().lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}, got {mode!r}.")
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        default_prefix = BGE_QUERY_PREFIX if "bge" in embedding_model.lower() and "-en-" in embedding_model.lower() else ""
        settings = cls(
            llm_api_key=os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY") or None,
            llm_model=os.getenv("GROQ_MODEL") or os.getenv("XAI_MODEL", "openai/gpt-oss-20b"),
            llm_base_url=(
                os.getenv("GROQ_BASE_URL") or os.getenv("XAI_BASE_URL", "https://api.groq.com/openai/v1")
            ).rstrip("/"),
            embedding_model=embedding_model,
            embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", default_prefix),
            data_dir=Path(os.getenv("RAG_DATA_DIR", BACKEND_DIR / "data")).resolve(),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", BACKEND_DIR / "storage")).resolve(),
            chunk_max_words=_env_int("CHUNK_MAX_WORDS", 250, minimum=50),
            chunk_overlap_words=_env_int("CHUNK_OVERLAP_WORDS", 40, minimum=0),
            retrieval_mode=mode,
            top_k=_env_int("TOP_K", 5),
            candidate_k=_env_int("CANDIDATE_K", 20),
            rerank_pool=_env_int("RERANK_POOL", 12),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            min_similarity=_env_float("MIN_SIMILARITY", 0.40, 0.0, 1.0),
            history_turns=_env_int("HISTORY_TURNS", 6, minimum=0),
        )
        if settings.chunk_overlap_words >= settings.chunk_max_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_MAX_WORDS.")
        return settings
