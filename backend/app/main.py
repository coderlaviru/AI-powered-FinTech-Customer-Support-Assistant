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
    history: list[Turn] = Field(default_factory=list, max_length=20)  # pyright: ignore[reportUnknownVariableType]
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
            detail="The configured LLM provider quota is exhausted or rate-limited. Check the API key's credits and limits.",
        ) from exc
    except openai.AuthenticationError as exc:
        logger.warning("LLM authentication failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="The LLM provider rejected the API key configured in backend/.env."
        ) from exc
    except openai.APIConnectionError as exc:
        logger.warning("Could not reach the LLM API: %s", exc)
        raise HTTPException(status_code=503, detail="Could not reach the configured LLM provider. Check the network connection.") from exc
    except openai.APIStatusError as exc:
        logger.warning("LLM API returned HTTP %s: %s", exc.status_code, exc)
        provider_message = ""
        if isinstance(exc.body, dict):
            error = exc.body.get("error") # type: ignore
            if isinstance(error, dict):
                for key in ("message", "detail"):
                    value = error.get(key) # type: ignore
                    if isinstance(value, str) and value.strip():
                        provider_message = value.strip()
                        break
            elif isinstance(error, str):
                provider_message = error.strip()
            if not provider_message:
                for key in ("message", "detail"):
                    value = exc.body.get(key) # type: ignore
                    if isinstance(value, str) and value.strip():
                        provider_message = value.strip()
                        break
        if not provider_message:
            provider_message = exc.message.strip()
            if provider_message.startswith("Error code:") and " - " in provider_message:
                provider_message = provider_message.split(" - ", 1)[1]
        provider_message = provider_message[:300]
        detail = f"The configured LLM provider rejected the request (HTTP {exc.status_code})."
        if provider_message:
            detail += f" Provider error: {provider_message}"
        else:
            detail += " Check GROQ_MODEL, GROQ_BASE_URL, and GROQ_API_KEY in backend/.env."
        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from exc
    except Exception as exc:
        logger.exception("Failed to answer a question")
        raise HTTPException(
            status_code=503, detail="The assistant could not process this question. Check the server logs."
        ) from exc

    return ChatResponse(**asdict(result))
