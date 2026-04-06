"""
Evaluation module.
Runs a simple evaluation pipeline against a test dataset.
Metrics:
- Retrieval recall: did the expected source appear in retrieved chunks?
- Answer relevance: keyword overlap between expected answer and LLM response
- Latency: time taken per query
"""

import json
import time
import os
from app.logger import get_logger

logger = get_logger("evaluator")

# Default test dataset path
EVAL_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "eval_dataset.json"
)


def _keyword_recall(expected: str, actual: str) -> float:
    """Fraction of expected answer keywords found in actual response."""
    import re
    exp_words = set(re.findall(r'\w+', expected.lower()))
    act_words = set(re.findall(r'\w+', actual.lower()))
    if not exp_words:
        return 0.0
    return round(len(exp_words & act_words) / len(exp_words), 3)


def _source_recall(expected_source: str, retrieved_chunks: list[dict]) -> bool:
    """Check if the expected source document appears in retrieved chunks."""
    sources = [c.get("source", "") for c in retrieved_chunks]
    return any(expected_source in s for s in sources)


def run_evaluation(pipeline_fn, dataset_path: str = EVAL_DATASET_PATH) -> dict:
    """
    Run evaluation against a JSON test dataset.

    Dataset format:
    [
      {
        "query": "What is RAG?",
        "expected_answer": "RAG stands for Retrieval Augmented Generation...",
        "expected_source": "rag_intro.txt"
      },
      ...
    ]

    pipeline_fn signature: (query: str) -> {"answer": str, "chunks": list[dict], "latency": float}
    """
    if not os.path.exists(dataset_path):
        logger.warning(f"Eval dataset not found at {dataset_path}. Skipping evaluation.")
        return {"error": "Dataset not found", "path": dataset_path}

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    total_recall = 0.0
    total_source_hit = 0
    total_latency = 0.0

    logger.info(f"Running evaluation on {len(dataset)} test cases...")

    for i, case in enumerate(dataset, 1):
        query = case["query"]
        expected_answer = case.get("expected_answer", "")
        expected_source = case.get("expected_source", "")

        start = time.time()
        output = pipeline_fn(query)
        latency = round(time.time() - start, 3)

        answer = output.get("answer", "")
        chunks = output.get("chunks", [])

        recall = _keyword_recall(expected_answer, answer)
        source_hit = _source_recall(expected_source, chunks) if expected_source else None

        result = {
            "case": i,
            "query": query,
            "answer_recall": recall,
            "source_hit": source_hit,
            "latency_s": latency
        }
        results.append(result)
        total_recall += recall
        if source_hit:
            total_source_hit += 1
        total_latency += latency

        logger.info(
            f"Case {i}/{len(dataset)} | recall={recall} | "
            f"source_hit={source_hit} | latency={latency}s"
        )

    n = len(dataset)
    summary = {
        "total_cases": n,
        "avg_answer_recall": round(total_recall / n, 3) if n else 0,
        "source_hit_rate": round(total_source_hit / n, 3) if n else 0,
        "avg_latency_s": round(total_latency / n, 3) if n else 0,
        "results": results
    }

    logger.info(
        f"Evaluation complete | avg_recall={summary['avg_answer_recall']} | "
        f"source_hit_rate={summary['source_hit_rate']} | avg_latency={summary['avg_latency_s']}s"
    )
    return summary
