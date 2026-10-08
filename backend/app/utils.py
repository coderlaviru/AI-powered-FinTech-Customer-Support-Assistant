"""Text normalisation helpers shared by ingestion, retrieval and evaluation."""

from __future__ import annotations

import re

_WS_RUN = re.compile(r"[ \t\f\v\u00a0]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
_TOKEN = re.compile(r"[a-z0-9₹]+(?:[-./][a-z0-9]+)*")
_PAGE_FOOTER = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+)\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalise extracted text without changing its meaning."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if not _PAGE_FOOTER.match(line)]
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def tokenize(text: str) -> list[str]:
    """Lower-case tokens for BM25. Identifiers such as ``met-pl-1301`` stay whole and
    are also indexed by their parts so partial queries (``pl-1301``, ``1301``) still match."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        parts = re.split(r"[-./]", token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part)
    return tokens


def word_count(text: str) -> int:
    return len(text.split())
