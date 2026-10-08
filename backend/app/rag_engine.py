"""Grounded RAG engine.

User query -> (follow-up rewrite) -> retrieval -> relevance gate -> grounded prompt -> LLM
-> citation validation -> answer with verified sources.

Hallucination mitigation is layered:
  1. Relevance gate: when even the best chunk is too dissimilar, the LLM is never called.
  2. Grounding prompt: answer only from numbered passages, cite every claim, otherwise
     emit the NOT_FOUND sentinel.
  3. Post-checks: every citation marker is validated against the retrieved set; answers whose
     citations cannot be verified are flagged to the caller.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .config import Settings
from .embeddings import LocalEmbedder
from .index_store import load_or_build_collection
from .llm import ChatLLM
from .retrieval import CrossEncoderReranker, Hit, HybridRetriever

logger = logging.getLogger(__name__)

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = (
    "I couldn't find this information in the FinBase knowledge base, so I can't answer it reliably. "
    "Please contact FinBase support for help with this question."
)

_MAX_QUESTION_CHARS = 300
_EXCERPT_CHARS = 700

_CITE_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")

_INVENTORY_PATTERNS = [
    re.compile(r"\b(list|name|enumerate|show)\b.{0,40}\b(all|every|each|available)\b.{0,40}\b(polic(?:y|ies)|documents?|guidelines)\b", re.I),
    re.compile(r"\bwhich\b.{0,25}\b(polic(?:y|ies)|documents|guidelines)\b.{0,40}\b(available|do you have|are there|covered|included|in the (?:knowledge base|kb))\b", re.I),
    re.compile(r"\bwhat\b.{0,25}\b(polic(?:y|ies)|documents|topics)\b.{0,30}\b(available|do you have|are there|covered|can you (?:answer|help)|in the (?:knowledge base|kb))", re.I),
    re.compile(r"\bhow many\b.{0,25}\b(polic(?:y|ies)|documents)\b", re.I),
]


class ConfigurationError(RuntimeError):
    """Raised when the engine cannot start because of missing configuration."""


class TextModel(Protocol):
    def complete(self, prompt: str, **kwargs): ...


@dataclass(frozen=True)
class Source:
    chunk_id: str
    file_name: str
    document_title: str
    section: str
    pages: str
    label: str
    excerpt: str
    similarity: float
    cited: bool


@dataclass
class ChatResult:
    answer: str
    sources: list[Source]
    refused: bool
    route: str  # rag | catalog | gate_refusal | llm_refusal
    retrieval_score: float
    citations_verified: bool
    standalone_question: str
    retrieval_mode: str
    timings: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------ helpers


def citation_label(meta: dict) -> str:
    title = meta.get("document_title", meta.get("file_name", "Document"))
    if meta.get("faq_id"):
        where = f"FAQ {meta['faq_id']} (Section {meta.get('section_id', '?')})"
    else:
        where = meta.get("section", "")
    return f"{title} \u2014 {where}, p. {meta.get('pages', '?')}"


def is_inventory_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in _INVENTORY_PATTERNS)


def process_citations(answer: str, passage_count: int) -> tuple[str, list[int], bool]:
    """Validate ``[S#]`` markers. Returns (cleaned answer, cited ids in order, any_invalid)."""
    cited: list[int] = []
    invalid = False

    def replace(match: re.Match) -> str:
        nonlocal invalid
        ids = [int(token[1:]) for token in re.split(r"\s*[,;]\s*", match.group(1))]
        valid = [i for i in ids if 1 <= i <= passage_count]
        if len(valid) != len(ids):
            invalid = True
        for i in valid:
            if i not in cited:
                cited.append(i)
        return "".join(f"[S{i}]" for i in valid)

    cleaned = _CITE_GROUP.sub(replace, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), cited, invalid


def _excerpt(text: str) -> str:
    lines = text.split("\n", 1)
    body = lines[1] if len(lines) > 1 else lines[0]
    return body if len(body) <= _EXCERPT_CHARS else body[: _EXCERPT_CHARS - 1].rstrip() + "\u2026"


def _format_history(history: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{turn['role'].capitalize()}: {turn['content'].strip()}" for turn in history)


def build_answer_prompt(question: str, hits: Sequence[Hit]) -> str:
    passages = "\n\n".join(f"[S{i}] {hit.text}" for i, hit in enumerate(hits, start=1))
    return (
        "You are the FinBase customer-support assistant. Answer the QUESTION using ONLY the numbered "
        "CONTEXT passages below.\n\n"
        "Rules:\n"
        "1. Use only facts stated in CONTEXT. Never use outside knowledge and never guess.\n"
        "2. Cite the supporting passage(s) after every sentence that states a fact, using the markers "
        "[S1], [S2], ... exactly as numbered. Do not cite section numbers that appear inside the passage "
        "text; only use the [S#] markers.\n"
        f"3. If CONTEXT does not contain the answer, reply with exactly {NOT_FOUND_TOKEN} and nothing else.\n"
        "4. If only part of the question is answerable, answer that part and say which part is not covered "
        "in the documents.\n"
        "5. Copy amounts, percentages, limits and time periods exactly as written, including currency symbols.\n"
        "6. If the passages say a service is not offered, state that clearly.\n"
        "7. Be concise: at most 120 words unless a table is clearly needed.\n\n"
        f"CONTEXT:\n{passages}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def build_rewrite_prompt(question: str, history: Sequence[dict[str, str]]) -> str:
    return (
        "Rewrite the FOLLOW-UP question as one standalone question that can be understood without the "
        "conversation. Keep every specific product, number and term; resolve pronouns and omissions from the "
        "conversation. If it is already standalone, return it unchanged. Output only the question.\n\n"
        f"CONVERSATION:\n{_format_history(history)}\n\n"
        f"FOLLOW-UP: {question}\n\n"
        "STANDALONE QUESTION:"
    )


# ------------------------------------------------------------------------------ engine


class RagEngine:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: TextModel) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    def generate(self, prompt: str) -> str:
        response = self.llm.complete(prompt)
        return (getattr(response, "text", None) or "").strip()

    def rewrite_question(self, question: str, history: Sequence[dict[str, str]]) -> str:
        if not history:
            return question
        try:
            rewritten = self.generate(build_rewrite_prompt(question, history))
        except Exception:  # rewriting is an optimisation; never fail the request because of it
            logger.exception("Question rewriting failed; using the original question")
            return question
        rewritten = rewritten.splitlines()[0].strip().strip('"').strip() if rewritten else ""
        if not rewritten or len(rewritten) > _MAX_QUESTION_CHARS:
            return question
        return rewritten

    def _catalog_answer(self) -> str:
        docs = self.retriever.documents()
        lines = [f"The FinBase knowledge base contains {len(docs)} documents:"]
        for number, doc in enumerate(docs, start=1):
            code = f" (code {doc['code']})" if doc["code"] else ""
            lines.append(f"{number}. {doc['title']}{code}")
        return "\n".join(lines)

    def answer(
        self,
        question: str,
        history: Sequence[dict[str, str]] | None = None,
        mode: str | None = None,
    ) -> ChatResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        turns = self.settings.history_turns
        history = list(history or [])[-turns:] if turns else []
        timings: dict[str, float] = {}

        started = time.perf_counter()
        standalone = self.rewrite_question(question, history)
        timings["rewrite"] = time.perf_counter() - started

        if is_inventory_question(standalone):
            return ChatResult(
                answer=self._catalog_answer(),
                sources=[],
                refused=False,
                route="catalog",
                retrieval_score=1.0,
                citations_verified=True,
                standalone_question=standalone,
                retrieval_mode="catalog",
                timings=timings,
            )

        started = time.perf_counter()
        retrieval = self.retriever.retrieve(standalone, mode=mode)
        timings["retrieve"] = time.perf_counter() - started

        def refusal(route: str) -> ChatResult:
            return ChatResult(
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                refused=True,
                route=route,
                retrieval_score=retrieval.top_similarity,
                citations_verified=True,
                standalone_question=standalone,
                retrieval_mode=retrieval.mode,
                timings=timings,
            )

        if retrieval.top_similarity < self.settings.min_similarity or not retrieval.hits:
            logger.info(
                "Gate refusal (top similarity %.3f < %.3f)", retrieval.top_similarity, self.settings.min_similarity
            )
            return refusal("gate_refusal")

        started = time.perf_counter()
        raw = self.generate(build_answer_prompt(standalone, retrieval.hits))
        timings["generate"] = time.perf_counter() - started

        if not raw or raw.strip().strip(".").upper().startswith(NOT_FOUND_TOKEN):
            return refusal("llm_refusal")

        cleaned, cited_ids, invalid = process_citations(raw, len(retrieval.hits))
        verified = bool(cited_ids) and not invalid
        if not verified:
            logger.warning("Answer citations could not be verified (cited=%s, invalid=%s)", cited_ids, invalid)

        cited_set = set(cited_ids)
        positions = [*cited_ids, *(p for p in range(1, len(retrieval.hits) + 1) if p not in cited_set)]
        sources = []
        for position in positions:
            hit = retrieval.hits[position - 1]
            sources.append(
                Source(
                    chunk_id=hit.chunk_id,
                    file_name=hit.metadata["file_name"],
                    document_title=hit.metadata["document_title"],
                    section=hit.metadata["section"],
                    pages=str(hit.metadata["pages"]),
                    label=citation_label(hit.metadata),
                    excerpt=_excerpt(hit.text),
                    similarity=round(hit.dense_similarity, 4),
                    cited=position in cited_set,
                )
            )
        logger.info(
            "route=rag mode=%s top_sim=%.3f cited=%s verified=%s timings=%s",
            retrieval.mode,
            retrieval.top_similarity,
            cited_ids,
            verified,
            {k: round(v, 2) for k, v in timings.items()},
        )
        return ChatResult(
            answer=cleaned,
            sources=sources,
            refused=False,
            route="rag",
            retrieval_score=retrieval.top_similarity,
            citations_verified=verified,
            standalone_question=standalone,
            retrieval_mode=retrieval.mode,
            timings=timings,
        )


# ------------------------------------------------------------------------------ factory

_engine: RagEngine | None = None
_engine_lock = threading.Lock()


def build_engine(settings: Settings) -> RagEngine:
    if not settings.llm_api_key:
        raise ConfigurationError("XAI_API_KEY is not set. Copy backend/.env.example to backend/.env and add it.")
    embedder = LocalEmbedder(settings.embedding_model, query_prefix=settings.embedding_query_prefix)
    llm = ChatLLM(settings.llm_api_key, settings.llm_model, settings.llm_base_url)
    collection = load_or_build_collection(settings, embedder)
    retriever = HybridRetriever(collection, embedder, settings, reranker=CrossEncoderReranker(settings.reranker_model))
    return RagEngine(settings, retriever, llm)


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = build_engine(Settings.from_env())
    return _engine
