import json

import evaluate as ev
from conftest import ScriptedLLM

LOC = {"file": "a.pdf", "section": "6.2"}


def meta(section, file="a.pdf", faq=""):
    return {"file_name": file, "section_id": section, "faq_id": faq}


def test_retrieval_scores_hit_recall_mrr():
    gold = [[LOC]]
    scores = ev.retrieval_scores([meta("1"), meta("2"), meta("6.2")], gold)
    assert scores["hit@1"] == 0.0 and scores["hit@3"] == 1.0 and scores["mrr"] == 1 / 3


def test_multi_group_recall_is_fractional():
    gold = [[LOC], [{"file": "a.pdf", "section": "9"}]]
    scores = ev.retrieval_scores([meta("6.2"), meta("1")], gold)
    assert scores["recall@3"] == 0.5 and scores["hit@3"] == 0.0


def test_faq_location_requires_matching_faq_id():
    loc = {"file": "a.pdf", "section": "23", "faq": "Q001"}
    assert ev.location_matches(meta("23", faq="Q001"), loc)
    assert not ev.location_matches(meta("23", faq="Q002"), loc)


def test_facts_present_normalises_commas_case_and_alternatives():
    assert ev.facts_present("It is \u20b91,00,000 per day", ["\u20b91,00,000"])
    assert ev.facts_present("rate is 1.5%", ["1.50%|1.5%"])
    assert not ev.facts_present("rate is 2%", ["1.5%"])


def test_parse_json_is_tolerant():
    assert ev.parse_json('Sure: {"correct": true, "reason": "x"} done') == {"correct": True, "reason": "x"}
    assert ev.parse_json("no json") is None


class JudgeLLM(ScriptedLLM):
    def complete(self, prompt, **kw):
        if "You grade" in prompt:
            self.default = '{"correct": true, "reason": "ok"}'
        elif "Check whether each factual claim" in prompt:
            self.default = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
        elif "STANDALONE QUESTION" in prompt:
            self.default = "standalone"
        else:
            self.default = "Answer [S1]."
        return super().complete(prompt, **kw)


def test_full_report_runs_offline_with_all_sections(make_engine):
    engine = make_engine(llm=JudgeLLM())
    items = ev.load_eval_set()[:4] + [i for i in ev.load_eval_set() if i["category"] == "unanswerable"][:2]
    report = ev.build_report(engine, items, ["dense", "hybrid"], "hybrid", use_judge=True)

    assert set(report["retrieval_ablation"]) == {"dense", "hybrid"}
    summary = report["pipeline"]["summary"]
    assert summary["answerable_count"] == 4 and summary["unanswerable_count"] == 2
    assert summary["judge_correctness"] == 1.0 and summary["groundedness"] == 0.5
    assert report["pipeline"]["failed"] == []
    assert report["similarity_calibration"]["answerable"] is not None
    markdown = ev.render_markdown(report)
    assert "Retrieval ablation" in markdown and "End-to-end" in markdown
    json.dumps(report)  # report must be JSON-serialisable


def test_a_failing_question_is_recorded_not_fatal(make_engine):
    class Exploding(JudgeLLM):
        def complete(self, prompt, **kw):
            raise RuntimeError("quota")

    engine = make_engine(llm=Exploding())
    run = ev.run_pipeline(engine, ev.load_eval_set()[:2], "dense", use_judge=False)
    assert run["failed"] == [i["id"] for i in ev.load_eval_set()[:2]]
