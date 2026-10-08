import pytest

from app.rag_engine import (
    NOT_FOUND_MESSAGE,
    build_answer_prompt,
    is_inventory_question,
    process_citations,
)
from conftest import ScriptedLLM


@pytest.mark.parametrize(
    "question",
    [
        "List all the policies you have",
        "Which documents are available in the knowledge base?",
        "What policies do you have?",
        "How many documents are covered?",
        "Name every policy document",
    ],
)
def test_inventory_questions_detected(question):
    assert is_inventory_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Does the policy apply to all borrowers?",
        "What is the name of the late fee policy?",
        "What are the late fee policies for credit cards?",
        "Show me the foreclosure charge",
        "What is the foreclosure charge?",
    ],
)
def test_normal_questions_are_not_treated_as_inventory(question):
    assert not is_inventory_question(question)


def test_process_citations_validates_and_normalises():
    cleaned, cited, invalid = process_citations("A [S2, S1]. B [S9].", passage_count=3)
    assert cleaned == "A [S2][S1]. B ."
    assert cited == [2, 1] and invalid


def test_grounded_answer_returns_verified_sources(make_engine):
    engine = make_engine(llm=ScriptedLLM(["The charge is 3% [S1]."]))
    result = engine.answer("What is the foreclosure charge for a personal loan before 24 months?")
    assert not result.refused and result.route == "rag" and result.citations_verified
    assert result.sources[0].cited and not all(s.cited for s in result.sources)
    assert "Section" in result.sources[0].label or "FAQ" in result.sources[0].label


def test_not_found_sentinel_returns_refusal_without_sources(make_engine):
    engine = make_engine(llm=ScriptedLLM(["NOT_FOUND"]))
    result = engine.answer("What is the foreclosure charge?")
    assert result.refused and result.route == "llm_refusal"
    assert result.answer == NOT_FOUND_MESSAGE and result.sources == []


def test_gate_refuses_without_calling_the_llm(make_engine):
    llm = ScriptedLLM()
    engine = make_engine(llm=llm, min_similarity=0.99)
    result = engine.answer("What is the capital of France?")
    assert result.refused and result.route == "gate_refusal"
    assert llm.prompts == []


def test_uncited_or_invalid_citations_are_flagged(make_engine):
    uncited = make_engine(llm=ScriptedLLM(["The charge is 3%."])).answer("foreclosure charge")
    invalid = make_engine(llm=ScriptedLLM(["The charge is 3% [S42]."])).answer("foreclosure charge")
    assert not uncited.citations_verified and not invalid.citations_verified


def test_follow_up_is_rewritten_using_history(make_engine):
    llm = ScriptedLLM(["What is the foreclosure charge after 24 months?", "It is 1.5% [S1]."])
    engine = make_engine(llm=llm)
    history = [
        {"role": "user", "content": "What is the foreclosure charge before 24 months?"},
        {"role": "assistant", "content": "3% of outstanding principal."},
    ]
    result = engine.answer("And after 24 months?", history=history)
    assert result.standalone_question == "What is the foreclosure charge after 24 months?"
    assert "FOLLOW-UP: And after 24 months?" in llm.prompts[0]
    assert "QUESTION: What is the foreclosure charge after 24 months?" in llm.prompts[1]


def test_rewrite_failure_falls_back_to_original_question(make_engine):
    class Boom(ScriptedLLM):
        def complete(self, prompt, **kw):
            if "STANDALONE QUESTION" in prompt:
                raise RuntimeError("down")
            return super().complete(prompt, **kw)

    engine = make_engine(llm=Boom(["ok [S1]"]))
    result = engine.answer("And after 24 months?", history=[{"role": "user", "content": "x"}])
    assert result.standalone_question == "And after 24 months?"


def test_catalog_route_answers_from_index_metadata(make_engine):
    llm = ScriptedLLM()
    result = make_engine(llm=llm).answer("List all the policies you have")
    assert result.route == "catalog" and "6 documents" in result.answer and llm.prompts == []


def test_prompt_numbers_passages_and_states_rules(retriever):
    hits = retriever.retrieve("UPI limit", mode="dense").hits[:2]
    prompt = build_answer_prompt("What is the UPI limit?", hits)
    assert "[S1]" in prompt and "[S2]" in prompt and "NOT_FOUND" in prompt


def test_empty_question_rejected(make_engine):
    with pytest.raises(ValueError):
        make_engine().answer("   ")
