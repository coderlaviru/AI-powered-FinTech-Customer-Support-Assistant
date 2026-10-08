"""Integrity checks: every gold location exists in the corpus and contains the expected facts."""

import json
import re
from pathlib import Path

import pytest

from app.ingestion import build_chunks

BACKEND = Path(__file__).resolve().parents[1]
ITEMS = [json.loads(line) for line in (BACKEND / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"[\s,]+", "", text.lower())


@pytest.fixture(scope="module")
def corpus():
    chunks = []
    for pdf in sorted((BACKEND / "data").glob("*.pdf")):
        chunks.extend(build_chunks(pdf, 350, 40))
    return chunks


def matching(chunks, loc):
    return [
        c for c in chunks
        if c.metadata["file_name"] == loc["file"]
        and c.metadata["section_id"] == loc["section"]
        and (not loc.get("faq") or c.metadata["faq_id"] == loc["faq"])
    ]


def test_ids_unique_and_schema_complete():
    assert len({i["id"] for i in ITEMS}) == len(ITEMS)
    for item in ITEMS:
        assert item["question"].strip() and item["category"] in {"single", "multi", "followup", "unanswerable"}
        if item["category"] == "unanswerable":
            assert item["gold"] == []
        else:
            assert item["gold"] and item["must_contain"]
        if item["category"] == "followup":
            assert item["history"] and item["standalone"]


@pytest.mark.parametrize("item", [i for i in ITEMS if i["gold"]], ids=lambda i: i["id"])
def test_gold_locations_exist_and_contain_the_facts(item, corpus):
    available = []
    for group in item["gold"]:
        found = [c for loc in group for c in matching(corpus, loc)]
        assert found, f"{item['id']}: no chunk matches gold group {group}"
        available.extend(found)
    haystack = norm(" ".join(c.text for c in available))
    for fact in item["must_contain"]:
        assert any(norm(option) in haystack for option in fact.split("|")), f"{item['id']}: fact {fact!r} not in gold chunks"


@pytest.mark.parametrize(
    "term",
    ["home loan", "car loan", "ceo", "ifsc", "nri", "gold loan", "demat", "cheque book", "business loan", "france"],
)
def test_unanswerable_topics_are_really_absent_from_the_corpus(term, corpus):
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
    assert not any(pattern.search(c.text) for c in corpus)
