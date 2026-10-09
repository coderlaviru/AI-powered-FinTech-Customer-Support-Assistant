"""RAG evaluation harness.

Usage (from the ``backend`` directory):

    python evaluate.py                         # retrieval ablation + full pipeline metrics
    python evaluate.py --retrieval-only        # no generation calls (embeds queries only)
    python evaluate.py --pipeline-mode hybrid  # choose the mode used for the end-to-end run
    python evaluate.py --limit 10 --sleep 2    # smoke test / respect free-tier rate limits

What is measured
----------------
Retrieval (per retrieval mode, answerable questions, k = 1/3/5)
    Hit@k    every gold group has a matching chunk in the top k
    Recall@k fraction of gold groups covered in the top k
    MRR      reciprocal rank of the first chunk matching any gold location (top 5)
End-to-end (answerable questions)
    Fact match        all expected key facts occur in the answer (deterministic)
    Correctness       LLM judge compares the answer with the reference answer
    Groundedness      LLM judge: share of answer claims supported by the cited passages
    Citation validity every [S#] marker points to a retrieved passage and at least one is present
    Citation hit      at least one cited chunk is a gold location
    Citation precision share of cited chunks that are gold locations (strict)
    False refusal     answerable question that the assistant refused
End-to-end (unanswerable questions)
    Refusal rate      share correctly refused (gate or LLM)
Calibration
    Top dense similarity distribution for answerable vs unanswerable questions, and the
    MIN_SIMILARITY values that follow from it.

The judge uses the same Groq model as the generator, which can favour its own style; use the
numbers to compare pipeline variants rather than as absolute truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import RETRIEVAL_MODES  # noqa: E402
from app.rag_engine import RagEngine, get_engine  # noqa: E402

logger = logging.getLogger("evaluate")

EVAL_SET = Path(__file__).resolve().parent / "eval" / "eval_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "eval" / "results"
KS = (1, 3, 5)

# ------------------------------------------------------------------------------ data


def load_eval_set(path: Path = EVAL_SET) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise ValueError(f"{path} contains no evaluation items.")
    return items


def is_answerable(item: dict[str, Any]) -> bool:
    return item["category"] != "unanswerable"


# ------------------------------------------------------------------------------ matching


def location_matches(meta: dict, loc: dict[str, str]) -> bool:
    return (
        meta.get("file_name") == loc["file"]
        and str(meta.get("section_id")) == loc["section"]
        and (not loc.get("faq") or meta.get("faq_id") == loc["faq"])
    )


def group_matches(meta: dict, group: list[dict[str, str]]) -> bool:
    return any(location_matches(meta, loc) for loc in group)


def normalise(text: str) -> str:
    return re.sub(r"[\s,]+", "", text.lower())


def facts_present(answer: str, must_contain: Iterable[str]) -> bool:
    haystack = normalise(answer)
    return all(any(normalise(option) in haystack for option in fact.split("|")) for fact in must_contain)


# ------------------------------------------------------------------------------ retrieval metrics


def retrieval_scores(ranked_meta: list[dict], gold: list[list[dict[str, str]]], ks: Iterable[int] = KS) -> dict[str, float]:
    """Hit@k, Recall@k and MRR for one question given ranked chunk metadata."""
    out: dict[str, float] = {}
    for k in ks:
        top = ranked_meta[:k]
        covered = sum(1 for group in gold if any(group_matches(meta, group) for meta in top))
        out[f"recall@{k}"] = covered / len(gold)
        out[f"hit@{k}"] = float(covered == len(gold))
    out["mrr"] = 0.0
    for rank, meta in enumerate(ranked_meta, start=1):
        if any(group_matches(meta, group) for group in gold):
            out["mrr"] = 1.0 / rank
            break
    return out


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))], 3)


def run_retrieval_ablation(engine: RagEngine, items: list[dict[str, Any]], modes: Iterable[str]) -> dict[str, Any]:
    answerable = [item for item in items if is_answerable(item)]
    report: dict[str, Any] = {}
    for mode in modes:
        rows = []
        effective = mode
        for item in answerable:
            query = item.get("standalone", item["question"])
            result = engine.retriever.retrieve(query, mode=mode, top_k=max(KS))
            effective = result.mode
            scores = retrieval_scores([hit.metadata for hit in result.hits], item["gold"])
            rows.append({"id": item["id"], **scores})
        metric_names = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]
        report[mode] = {
            "effective_mode": effective,
            **{name: mean([row[name] for row in rows]) for name in metric_names},
            "per_question": rows,
        }
    return report


# ------------------------------------------------------------------------------ calibration


def similarity_calibration(engine: RagEngine, items: list[dict[str, Any]]) -> dict[str, Any]:
    ans, unans = [], []
    for item in items:
        query = item.get("standalone", item["question"])
        top = engine.retriever.retrieve(query, mode="dense", top_k=1).top_similarity
        (ans if is_answerable(item) else unans).append(top)
    result: dict[str, Any] = {
        "answerable": {"min": round(min(ans), 4), "mean": mean(ans), "max": round(max(ans), 4)} if ans else None,
        "unanswerable": {"min": round(min(unans), 4), "mean": mean(unans), "max": round(max(unans), 4)} if unans else None,
    }
    if ans and unans:
        candidates = sorted(set(ans + unans))
        best_t, best_score = candidates[0], -1.0
        for t in candidates:
            balanced = 0.5 * (sum(s >= t for s in ans) / len(ans) + sum(s < t for s in unans) / len(unans))
            if balanced > best_score:
                best_t, best_score = t, balanced
        result["never_block_valid_questions"] = round(min(ans) - 0.005, 3)
        result["best_balanced_threshold"] = round(best_t, 3)
        result["best_balanced_accuracy"] = round(best_score, 3)
        result["unanswerable_blocked_at_safe_threshold"] = round(
            sum(s < result["never_block_valid_questions"] for s in unans) / len(unans), 3
        )
    return result


# ------------------------------------------------------------------------------ judges

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def judge_correctness(engine: RagEngine, question: str, reference: str, answer: str) -> bool | None:
    prompt = (
        "You grade a customer-support answer against a reference answer.\n"
        'Reply with JSON only: {"correct": true|false, "reason": "<short>"}.\n'
        "correct is true only if the ANSWER states the facts in the REFERENCE (numbers, limits, periods, "
        "conditions) without contradicting them. Extra correct detail is fine. A refusal or 'not found' is incorrect.\n\n"
        f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}\n"
    )
    parsed = parse_json(engine.generate(prompt))
    if not parsed or not isinstance(parsed.get("correct"), bool):
        return None
    return parsed["correct"]


def judge_groundedness(engine: RagEngine, answer: str, passages: list[str]) -> float | None:
    if not passages:
        return None
    context = "\n\n".join(f"[P{i}] {text}" for i, text in enumerate(passages, start=1))
    prompt = (
        "Check whether each factual claim in the ANSWER is supported by the CONTEXT.\n"
        'Reply with JSON only: {"claims": [{"claim": "<text>", "supported": true|false}]}.\n'
        "Split the answer into atomic factual claims, ignore citation markers like [S1], and mark a claim "
        "supported only if the CONTEXT states it explicitly.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n"
    )
    parsed = parse_json(engine.generate(prompt))
    claims = parsed.get("claims") if parsed else None
    if not isinstance(claims, list) or not claims:
        return None
    flags = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
    return sum(flags) / len(flags) if flags else None


# ------------------------------------------------------------------------------ pipeline


def evaluate_item(engine: RagEngine, item: dict[str, Any], mode: str, use_judge: bool) -> dict[str, Any]:
    started = time.perf_counter()
    result = engine.answer(item["question"], history=item.get("history"), mode=mode)
    latency = time.perf_counter() - started
    row: dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": result.answer,
        "route": result.route,
        "refused": result.refused,
        "retrieval_score": round(result.retrieval_score, 4),
        "latency_s": round(latency, 3),
        "standalone_question": result.standalone_question,
    }
    if not is_answerable(item):
        return row

    cited = [s for s in result.sources if s.cited]
    metas = {s.chunk_id: engine.retriever._meta[s.chunk_id] for s in cited}
    row["fact_match"] = (not result.refused) and facts_present(result.answer, item["must_contain"])
    row["citation_valid"] = (not result.refused) and result.citations_verified
    row["citation_hit"] = any(group_matches(m, g) for m in metas.values() for g in item["gold"])
    gold_locations = [loc for group in item["gold"] for loc in group]
    row["citation_precision"] = (
        sum(any(location_matches(m, loc) for loc in gold_locations) for m in metas.values()) / len(metas) if metas else 0.0
    )
    row["cited_sources"] = [s.label for s in cited]
    if use_judge and not result.refused:
        row["judge_correct"] = judge_correctness(engine, item["question"], item["reference_answer"], result.answer)
        row["groundedness"] = judge_groundedness(
            engine, result.answer, [engine.retriever.chunk_text(s.chunk_id) for s in cited]
        )
    return row


def summarise_pipeline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    judged = [r["judge_correct"] for r in answerable if r.get("judge_correct") is not None]
    grounded = [r["groundedness"] for r in answerable if r.get("groundedness") is not None]
    latencies = [r["latency_s"] for r in rows]
    return {
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "false_refusal_rate": mean([float(r["refused"]) for r in answerable]),
        "fact_match_rate": mean([float(r["fact_match"]) for r in answerable]),
        "judge_correctness": mean([float(v) for v in judged]),
        "groundedness": mean(grounded),
        "citation_validity": mean([float(r["citation_valid"]) for r in answerable]),
        "citation_hit_rate": mean([float(r["citation_hit"]) for r in answerable]),
        "citation_precision": mean([r["citation_precision"] for r in answerable]),
        "refusal_rate_on_unanswerable": mean([float(r["refused"]) for r in unanswerable]),
        "latency_mean_s": mean(latencies),
        "latency_p95_s": percentile(latencies, 0.95),
    }


def run_pipeline(
    engine: RagEngine, items: list[dict[str, Any]], mode: str, use_judge: bool, sleep: float = 0.0
) -> dict[str, Any]:
    rows = []
    for position, item in enumerate(items, start=1):
        try:
            rows.append(evaluate_item(engine, item, mode, use_judge))
        except Exception as exc:  # one failing question must not discard the whole run
            logger.exception("Question %s failed", item["id"])
            rows.append({"id": item["id"], "category": item["category"], "question": item["question"], "error": str(exc)})
        logger.info("[%d/%d] %s", position, len(items), item["id"])
        if sleep:
            time.sleep(sleep)
    failed = [r for r in rows if "error" in r]
    completed = [r for r in rows if "error" not in r]
    return {"mode": mode, "failed": [r["id"] for r in failed], "summary": summarise_pipeline(completed), "rows": rows}


# ------------------------------------------------------------------------------ reporting


def _fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# RAG evaluation ({report['timestamp']})", ""]
    ablation = report.get("retrieval_ablation")
    if ablation:
        columns = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr"]
        lines += ["## Retrieval ablation (answerable questions)", "", "| mode | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for mode, metrics in ablation.items():
            label = mode if metrics["effective_mode"] == mode else f"{mode} (ran as {metrics['effective_mode']})"
            lines.append(f"| {label} | " + " | ".join(_fmt(metrics[c]) for c in columns) + " |")
        lines.append("")
    pipeline = report.get("pipeline")
    if pipeline:
        lines += [f"## End-to-end (mode: {pipeline['mode']})", "", "| metric | value |", "|---|---|"]
        lines += [f"| {key} | {_fmt(value)} |" for key, value in pipeline["summary"].items()]
        if pipeline["failed"]:
            lines += ["", f"Failed questions: {', '.join(pipeline['failed'])}"]
        lines.append("")
    calibration = report.get("similarity_calibration")
    if calibration:
        lines += ["## MIN_SIMILARITY calibration", "", "```json", json.dumps(calibration, indent=2), "```", ""]
    return "\n".join(lines)


def build_report(
    engine: RagEngine,
    items: list[dict[str, Any]],
    modes: list[str],
    pipeline_mode: str | None,
    use_judge: bool,
    sleep: float = 0.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "question_count": len(items),
        "retrieval_ablation": run_retrieval_ablation(engine, items, modes),
        "similarity_calibration": similarity_calibration(engine, items),
    }
    if pipeline_mode:
        report["pipeline"] = run_pipeline(engine, items, pipeline_mode, use_judge, sleep)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the FinTech RAG assistant.")
    parser.add_argument("--dataset", type=Path, default=EVAL_SET)
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--pipeline-mode", choices=RETRIEVAL_MODES, default="hybrid_rerank")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation and judging.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-judged metrics (fact match only).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between questions.")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    items = load_eval_set(args.dataset)
    if args.limit:
        items = items[: args.limit]

    engine = get_engine()
    report = build_report(
        engine,
        items,
        args.modes,
        None if args.retrieval_only else args.pipeline_mode,
        use_judge=not args.no_judge,
        sleep=args.sleep,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved: {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
