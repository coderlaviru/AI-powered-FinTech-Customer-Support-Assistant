"""Evaluate answer overlap and source retrieval against a JSONL test set."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from app.rag_engine import query


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(expected: str, actual: str) -> float:
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0

    expected_counts: dict[str, int] = {}
    actual_counts: dict[str, int] = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    for token in actual_tokens:
        actual_counts[token] = actual_counts.get(token, 0) + 1
    overlap = sum(min(count, actual_counts.get(token, 0)) for token, count in expected_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("question"), str)
            or not case["question"].strip()
        ):
            raise ValueError(f"Line {line_number} must be an object with a question string.")
        if "expected_answer" in case and not isinstance(case["expected_answer"], str):
            raise ValueError(f"Line {line_number} expected_answer must be a string.")
        if "expected_sources" in case and (
            not isinstance(case["expected_sources"], list)
            or not all(isinstance(source, str) for source in case["expected_sources"])
        ):
            raise ValueError(f"Line {line_number} expected_sources must be a list of strings.")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def evaluate(cases: list[dict[str, Any]]) -> dict[str, int | float]:
    answer_scores = []
    source_scores = []
    for case in cases:
        result = query(case["question"])
        expected_answer = case.get("expected_answer")
        if isinstance(expected_answer, str):
            answer_scores.append(token_f1(expected_answer, result["answer"]))

        expected_sources = case.get("expected_sources", [])
        if expected_sources:
            source_names = [source["file_name"].lower() for source in result["sources"]]
            source_scores.append(
                float(
                    any(
                        expected.lower() in source_name
                        for expected in expected_sources
                        for source_name in source_names
                    )
                )
            )

    metrics: dict[str, int | float] = {"cases": len(cases)}
    if answer_scores:
        metrics["mean_answer_token_f1"] = sum(answer_scores) / len(answer_scores)
    if source_scores:
        metrics["source_hit_rate"] = sum(source_scores) / len(source_scores)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSONL evaluation cases file")
    args = parser.parse_args()
    try:
        metrics = evaluate(load_cases(args.cases))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
