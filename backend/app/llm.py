"""Chat-completion client for OpenAI-compatible APIs (Groq by default)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 1500
REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 3


@dataclass(frozen=True)
class Completion:
    text: str


class ChatLLM:
    """Single-turn text completion with deterministic decoding.

    Rate-limit, timeout and 5xx errors are retried with backoff by the OpenAI SDK; other API
    errors propagate so the API layer can report them.
    """

    def __init__(self, api_key: str, model: str, base_url: str, client: OpenAI | None = None) -> None:
        self._model = model
        self._client = client or OpenAI(
            api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES
        )

    def complete(self, prompt: str, **_: object) -> Completion:
        response = self._client.chat.completions.create(  # type: ignore
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice and choice.message else ""
        if choice is not None and choice.finish_reason == "length":
            logger.warning("Completion hit the %d-token output limit and may be truncated", MAX_OUTPUT_TOKENS)
        return Completion(text=text.strip())
