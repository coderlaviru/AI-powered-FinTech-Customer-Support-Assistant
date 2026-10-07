#backend/app/rag _engine.py
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from pypdf import PdfReader

from app.utils import clean_text

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BACKEND_DIR / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

GROUNDING_PROMPT = PromptTemplate(
    "You are a financial customer-support assistant. Answer only using the provided "
    "context. If the context does not contain the answer, say that you could not find "
    "the information in the provided documents. Do not make up financial terms, "
    "numbers, or policies. Cite supporting sources using the provided citation markers.\n\n"
    "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
)

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BACKEND_DIR / "data")).resolve()
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BACKEND_DIR / "storage")).resolve()
MANIFEST_PATH = INDEX_DIR / "source_fingerprint.json"

_lock = threading.Lock()
_query_engine: CitationQueryEngine | None = None


def _source_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"gemini:{GEMINI_EMBEDDING_MODEL}".encode("utf-8"))
    for path in files:
        digest.update(path.relative_to(DATA_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _is_policy_inventory_question(question: str) -> bool:
    words = set(re.findall(r"\w+", question.lower()))
    asks_for_inventory = bool(words & {"all", "every", "list", "name", "enumerate"})
    mentions_policy = bool(words & {"policy", "policies"})
    return asks_for_inventory and mentions_policy


def _list_policy_documents() -> dict[str, Any]:
    files = sorted(path for path in DATA_DIR.rglob("*.pdf") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No PDF documents found in {DATA_DIR}")

    sources = []
    titles = []
    for path in files:
        try:
            reader = PdfReader(str(path))
            first_page = reader.pages[0].extract_text() if reader.pages else ""
        except Exception as exc:
            logger.exception("Failed to read the first page of %s", path.name)
            raise ValueError(f"Could not read policy document {path.name}.") from exc

        if not first_page or not first_page.strip():
            raise ValueError(f"Could not extract a title from policy document {path.name}.")
        title_parts = first_page.split("Document Code:", maxsplit=1)
        title_text = title_parts[0] if len(title_parts) > 1 else first_page.splitlines()[0]
        title = clean_text(" ".join(title_text.split()))
        if not title:
            title = path.stem
        titles.append(title)
        sources.append(
            {
                "file_name": path.name,
                "page": "1",
                "score": None,
                "excerpt": title,
            }
        )

    answer = "The policy documents in the knowledge base are:\n" + "\n".join(
        f"- {title} ({source['file_name']})"
        for title, source in zip(titles, sources)
    )
    return {"answer": answer, "sources": sources}


def _configure_models() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to backend/.env.")

    Settings.llm = GoogleGenAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=0,
    )
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=GEMINI_EMBEDDING_MODEL,
        api_key=api_key,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "768")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
    )


def _load_or_build_index(files: list[Path]) -> VectorStoreIndex:
    fingerprint = _source_fingerprint(files)
    if INDEX_DIR.is_dir() and MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == fingerprint:
                storage = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
                return load_index_from_storage(storage)
        except Exception as e:
            logger.warning(f"Could not load index from storage, rebuilding: {e}")

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=True,
        required_exts=[".pdf"],
        filename_as_id=True,
    ).load_data()

    for document in documents:
        document.set_content(clean_text(document.text))

    index = VectorStoreIndex.from_documents(documents)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    temporary_manifest = MANIFEST_PATH.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2),
        encoding="utf-8",
    )
    temporary_manifest.replace(MANIFEST_PATH)
    return index


def get_query_engine() -> CitationQueryEngine:
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    with _lock:
        if _query_engine is not None:
            return _query_engine
        if not DATA_DIR.is_dir():
            raise FileNotFoundError(f"PDF data directory does not exist: {DATA_DIR}")
        files = sorted(path for path in DATA_DIR.rglob("*.pdf") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No PDF documents found in {DATA_DIR}")

        _configure_models()
        index = _load_or_build_index(files)
        _query_engine = CitationQueryEngine.from_args(
            index,
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "4")),
            citation_chunk_size=512,
            citation_qa_template=GROUNDING_PROMPT,
        )
        return _query_engine


def query(question: str) -> dict[str, Any]:
    if _is_policy_inventory_question(question):
        return _list_policy_documents()

    response = get_query_engine().query(question)
    sources = []

    for source_node in response.source_nodes:
        node = source_node.node
        metadata = node.metadata or {}

        # CitationQueryEngine generates internal virtual nodes.
        # We fetch native file fields out safely to meet Pydantic validations.
        raw_path = metadata.get("file_path") or metadata.get("file_name") or "Unknown source"
        clean_file_name = Path(raw_path).name

        # Ensure we pass float formatting rules correctly
        score = float(source_node.score) if source_node.score is not None else None

        sources.append(
            {
                "file_name": clean_file_name,
                "page": metadata.get("page_label"),
                "score": score,
                "excerpt": node.get_content()[:500],
            }
        )

    return {"answer": str(response), "sources": sources}
