from fastapi.testclient import TestClient

import app.main as main
from conftest import ScriptedLLM


def test_chat_endpoint_returns_structured_response(make_engine, monkeypatch):
    engine = make_engine(llm=ScriptedLLM(["The limit is \u20b91,00,000 per day [S1]."]))
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={"question": "What is the daily UPI limit?", "history": [], "retrieval_mode": "dense"},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["answer"].endswith("[S1].") and body["citations_verified"] is True
    assert body["sources"][0]["cited"] is True and body["retrieval_mode"] == "dense"


def test_validation_and_configuration_errors(monkeypatch):
    from app.rag_engine import ConfigurationError

    def broken():
        raise ConfigurationError("XAI_API_KEY is not set.")

    monkeypatch.setattr(main, "get_engine", broken)
    with TestClient(main.app) as client:
        assert client.post("/chat", json={"question": "   "}).status_code == 422
        assert client.post("/chat", json={"question": "hi", "retrieval_mode": "x"}).status_code == 422
        response = client.post("/chat", json={"question": "hello"})
        assert response.status_code == 503 and "XAI_API_KEY" in response.json()["detail"]
        assert client.get("/health").json()["status"] == "ok"
