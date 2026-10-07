#backend/app/main.py
import logging
from fastapi import FastAPI, HTTPException
from google.genai.errors import ClientError
from pydantic import BaseModel, Field, field_validator
from app.rag_engine import query

# Configuration for absolute system monitoring transparency
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinTech Customer Support Assistant",
    description="Grounded financial-document question answering with source citations.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question


class Source(BaseModel):
    file_name: str
    page: str | None = None
    score: float | None = None
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "FinTech Customer Support Assistant API",
        "health": "/health",
        "docs": "/docs",
        "chat": "POST /chat",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        # Pass user string dynamically into your active LlamaIndex processing loop
        result = query(request.question.strip())

        # Verify the structure matches expected Pydantic keys exactly
        if not isinstance(result, dict) or "answer" not in result or "sources" not in result:
            raise ValueError("Invalid return dictionary format generated from the RAG engine loop.")

    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"Validation or data file anomaly encountered: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ClientError as exc:
        logger.warning("Gemini API request failed with status %s.", exc.code)
        detail = (
            "The Gemini API quota is exhausted or rate-limited. Check billing and "
            "usage limits for the API key configured in backend/.env."
            if exc.code == 429
            else f"The Gemini API rejected the request (HTTP {exc.code}). Check the API key and model configuration."
        )
        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from exc
    except Exception as exc:
        logger.exception("Failed to answer a question through the core pipeline execution layer")
        raise HTTPException(
            status_code=503,
            detail="The assistant could not process this question. Check the server logs.",
        ) from exc

    return ChatResponse(**result)
