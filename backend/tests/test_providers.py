import httpx
import numpy as np
import openai
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import BGE_QUERY_PREFIX, Settings
from app.embeddings import LocalEmbedder
from app.llm import ChatLLM


class FakeCompletions:
    def __init__(self, content="  Answer [S1].  ", finish="stop", raises=None):
        self.calls = []
        self._content, self._finish, self._raises = content, finish, raises

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        message = type("M", (), {"content": self._content})()
        choice = type("C", (), {"message": message, "finish_reason": self._finish})()
        return type("R", (), {"choices": [choice]})()


def fake_client(completions):
    return type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()


def test_chat_llm_sends_deterministic_request_and_strips_text():
    completions = FakeCompletions()
    llm = ChatLLM("k", "grok-test", "https://api.groq.com/openai/v1", client=fake_client(completions))
    assert llm.complete("hello").text == "Answer [S1]."
    call = completions.calls[0]
    assert call["model"] == "grok-test" and call["temperature"] == 0.0
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["max_completion_tokens"] == 1500


def test_chat_llm_handles_empty_content():
    llm = ChatLLM("k", "m", "u", client=fake_client(FakeCompletions(content=None)))
    assert llm.complete("x").text == ""


class StubModel:
    max_seq_length = 8

    class tokenizer:  # noqa: N801
        @staticmethod
        def encode(text, add_special_tokens=True):
            return text.split()

    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.seen.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts])


def test_local_embedder_prefixes_only_queries_and_returns_lists():
    model = StubModel()
    embedder = LocalEmbedder("m", query_prefix="Q: ", model_factory=lambda _: model)
    assert embedder.get_text_embedding_batch(["abc", "de"]) == [[3.0, 1.0], [2.0, 1.0]]
    assert embedder.get_query_embedding("hi") == [5.0, 1.0]
    assert model.seen[0] == ["abc", "de"] and model.seen[1] == ["Q: hi"]


def test_local_embedder_warns_once_about_truncation(caplog):
    embedder = LocalEmbedder("m", model_factory=lambda _: StubModel())
    with caplog.at_level("WARNING"):
        embedder.get_text_embedding_batch(["word " * 20])
        embedder.get_text_embedding_batch(["word " * 20])
    assert sum("truncated" in r.message for r in caplog.records) == 1


def test_settings_defaults_and_bge_prefix(monkeypatch):
    for name in (
        "EMBEDDING_MODEL",
        "EMBEDDING_QUERY_PREFIX",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "GROQ_BASE_URL",
        "XAI_API_KEY",
        "XAI_MODEL",
        "XAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_query_prefix == ""

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    assert Settings.from_env().embedding_query_prefix == BGE_QUERY_PREFIX
    monkeypatch.delenv("EMBEDDING_MODEL")
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"

    monkeypatch.setenv("XAI_MODEL", "legacy-model")
    monkeypatch.setenv("XAI_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("XAI_API_KEY", "legacy-key")
    legacy_settings = Settings.from_env()
    assert legacy_settings.llm_model == "legacy-model"
    assert legacy_settings.llm_base_url == "https://legacy.example/v1"
    assert legacy_settings.llm_api_key == "legacy-key"

    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/")
    groq_settings = Settings.from_env()
    assert groq_settings.llm_api_key == "groq-key"
    assert groq_settings.llm_model == "openai/gpt-oss-120b"
    assert groq_settings.llm_base_url == "https://api.groq.com/openai/v1"

    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    assert Settings.from_env().embedding_query_prefix == ""
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    assert Settings.from_env().embedding_query_prefix == ""


def _request():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


@pytest.mark.parametrize(
    "error, fragment",
    [
        (openai.RateLimitError("limit", response=httpx.Response(429, request=_request()), body=None), "quota"),
        (openai.AuthenticationError("bad", response=httpx.Response(401, request=_request()), body=None), "API key"),
        (openai.APIConnectionError(request=_request()), "reach"),
        (openai.NotFoundError("nope", response=httpx.Response(404, request=_request()), body=None), "HTTP 404"),
    ],
)
def test_llm_errors_map_to_clear_503s(make_engine, monkeypatch, error, fragment):
    class Failing:
        def complete(self, prompt, **_):
            raise error

    engine = make_engine(llm=Failing())
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    with TestClient(main.app) as client:
        response = client.post("/chat", json={"question": "What is the foreclosure charge?"})
    assert response.status_code == 503 and fragment in response.json()["detail"]


def test_bad_request_includes_provider_error_message(make_engine, monkeypatch):
    error = openai.BadRequestError(
        "bad request",
        response=httpx.Response(
            400,
            request=_request(),
            json={"error": {"message": "Unknown model: grok-invalid"}},
        ),
        body={"error": {"message": "Unknown model: grok-invalid"}},
    )

    class Failing:
        def complete(self, prompt, **_):
            raise error

    engine = make_engine(llm=Failing())
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    with TestClient(main.app) as client:
        response = client.post("/chat", json={"question": "What is the foreclosure charge?"})

    assert response.status_code == 503
    assert "Unknown model: grok-invalid" in response.json()["detail"]


def test_bad_request_includes_top_level_provider_message(make_engine, monkeypatch):
    error = openai.BadRequestError(
        "Invalid parameter: max_tokens",
        response=httpx.Response(
            400,
            request=_request(),
            json={"message": "Invalid parameter: max_tokens"},
        ),
        body={"message": "Invalid parameter: max_tokens"},
    )

    class Failing:
        def complete(self, prompt, **_):
            raise error

    engine = make_engine(llm=Failing())
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    with TestClient(main.app) as client:
        response = client.post("/chat", json={"question": "What is the foreclosure charge?"})

    assert response.status_code == 503
    assert "Invalid parameter: max_tokens" in response.json()["detail"]
