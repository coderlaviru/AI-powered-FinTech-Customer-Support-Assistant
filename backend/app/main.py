"""FastAPI service exposing the grounded RAG assistant."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
import openai
from pydantic import BaseModel, Field, field_validator

from app.config import RETRIEVAL_MODES
from app.rag_engine import ConfigurationError, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_engine_state = {"ready": False}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the engine (loads or builds the index) so the first request is fast."""
    try:
        get_engine()
        _engine_state["ready"] = True
    except Exception:  # keep the API up so /health and /docs still explain what is wrong
        logger.exception("Engine warm-up failed; it will be retried on the first request")
    yield


app = FastAPI(
    title="FinTech Customer Support Assistant",
    description="Grounded financial-document question answering with verified source citations.",
    version="2.0.0",
    lifespan=lifespan,
)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    retrieval_mode: Literal["dense", "hybrid", "hybrid_rerank"] | None = Field(
        default=None, description=f"Override the retrieval strategy ({', '.join(RETRIEVAL_MODES)})."
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question


class Source(BaseModel):
    chunk_id: str
    file_name: str
    document_title: str
    section: str
    pages: str
    label: str
    excerpt: str
    similarity: float
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    refused: bool
    route: str
    retrieval_score: float
    citations_verified: bool
    standalone_question: str
    retrieval_mode: str
    timings: dict[str, float]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "FinTech Customer Support Assistant API",
        "health": "/health",
        "docs": "/docs",
        "chat": "POST /chat",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "engine_ready": _engine_state["ready"]}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        engine = get_engine()
        _engine_state["ready"] = True
        result = engine.answer(
            request.question,
            history=[turn.model_dump() for turn in request.history],
            mode=request.retrieval_mode,
        )
    except (FileNotFoundError, ConfigurationError) as exc:
        logger.warning("Configuration or data problem: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except openai.RateLimitError as exc:
        logger.warning("LLM rate limit or quota exhausted: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The xAI API quota is exhausted or rate-limited. Check credits and limits for the key in backend/.env.",
        ) from exc
    except openai.AuthenticationError as exc:
        logger.warning("LLM authentication failed: %s", exc)
        raise HTTPException(status_code=503, detail="xAI rejected the API key. Check XAI_API_KEY in backend/.env.") from exc
    except openai.APIConnectionError as exc:
        logger.warning("Could not reach the LLM API: %s", exc)
        raise HTTPException(status_code=503, detail="Could not reach the xAI API. Check the network connection.") from exc
    except openai.APIStatusError as exc:
        logger.warning("LLM API returned HTTP %s: %s", exc.status_code, exc)
        raise HTTPException(
            status_code=503,
            detail=f"The xAI API rejected the request (HTTP {exc.status_code}). Check XAI_MODEL and the API key.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to answer a question")
        raise HTTPException(
            status_code=503, detail="The assistant could not process this question. Check the server logs."
        ) from exc

    return ChatResponse(**asdict(result))
