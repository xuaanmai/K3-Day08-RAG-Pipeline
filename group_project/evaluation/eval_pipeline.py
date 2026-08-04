"""Evaluate the IELTS RAG pipeline with RAGAS and compare two retrievers.

Examples:
    python -m group_project.evaluation.eval_pipeline --limit 5
    python -m group_project.evaluation.eval_pipeline --limit 18

RAGAS invokes a judge LLM several times per question. Use ``--limit 5`` first
when the API account has a small rate limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

GOLDEN_DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.md")
DETAILS_PATH = Path(__file__).with_name("evaluation_details.json")
GENERATIONS_PATH = Path(__file__).with_name("generation_cache.json")
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def load_golden_dataset() -> list[dict]:
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if len(data) < 15:
        raise ValueError("Golden dataset must contain at least 15 cases")
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, 1):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Case {index} is missing: {sorted(missing)}")
    return data


def _run_pipeline(pipeline, item: dict, retrieval_mode: str) -> dict:
    function: Callable = getattr(pipeline, "generate_with_citation", pipeline)
    return function(item["question"], retrieval_mode=retrieval_mode)


def collect_samples(pipeline, dataset: list[dict], retrieval_mode: str) -> list[dict]:
    """Run generation once and retain all evidence needed to audit scores."""
    cache = {}
    if GENERATIONS_PATH.exists():
        try:
            cache = json.loads(GENERATIONS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    samples = []
    for index, item in enumerate(dataset, 1):
        print(f"[{retrieval_mode}] {index}/{len(dataset)}: {item['question']}")
        cache_key = f"{retrieval_mode}::{item['question']}"
        result = cache.get(cache_key)
        if result is None:
            result = _run_pipeline(pipeline, item, retrieval_mode)
            cache[cache_key] = result
            GENERATIONS_PATH.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        contexts = [
            str(source.get("content", ""))
            for source in result.get("sources", [])
            if source.get("content")
        ]
        samples.append(
            {
                "question": item["question"],
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
                "expected_context": item["expected_context"],
                "retrieval_source": result.get("retrieval_source", retrieval_mode),
            }
        )
    return samples


class _LocalEmbeddings:
    """LangChain embedding interface backed by the project's offline model."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        from src.task4_chunking_indexing import fallback_embedding, get_embedding_model

        model = get_embedding_model()
        if model is None:
            return fallback_embedding(text)
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _judge_models():
    """Create LangChain clients compatible with Gemini's OpenAI endpoint."""
    from langchain_openai import ChatOpenAI

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required to run RAGAS")
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm = ChatOpenAI(
        api_key=key,
        base_url=base_url,
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        max_retries=3,
    )
    return llm, _LocalEmbeddings()


def evaluate_with_ragas(pipeline, golden_dataset: list[dict], retrieval_mode="hybrid") -> dict:
    """Run the four required RAGAS metrics and return auditable row data."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    samples = collect_samples(pipeline, golden_dataset, retrieval_mode)
    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in samples],
            "answer": [row["answer"] for row in samples],
            "contexts": [row["contexts"] for row in samples],
            "ground_truth": [row["ground_truth"] for row in samples],
        }
    )
    llm, embeddings = _judge_models()
    evaluated = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    ).to_pandas().to_dict(orient="records")

    rows = []
    for sample, scores in zip(samples, evaluated):
        row = dict(sample)
        for metric in METRICS:
            value = scores.get(metric)
            row[metric] = None if value is None else float(value)
        rows.append(row)
    return {"framework": "RAGAS", "config": retrieval_mode, "rows": rows}


def _averages(result: dict) -> dict[str, float]:
    averages = {}
    for metric in METRICS:
        values = [row[metric] for row in result["rows"] if row.get(metric) is not None]
        averages[metric] = mean(values) if values else 0.0
    averages["average"] = mean(averages.values())
    return averages


def compare_configs(pipeline, golden_dataset: list[dict]) -> dict:
    """Compare hybrid + RRF against dense-only on the same dataset."""
    return {
        "hybrid": evaluate_with_ragas(pipeline, golden_dataset, "hybrid"),
        "dense_only": evaluate_with_ragas(pipeline, golden_dataset, "dense_only"),
    }


def export_results(comparison: dict, path: Path = RESULTS_PATH) -> None:
    """Write aggregate scores, worst cases and evidence-backed recommendations."""
    hybrid = _averages(comparison["hybrid"])
    dense = _averages(comparison["dense_only"])
    lines = [
        "# RAG Evaluation Results — IELTS Band Descriptors",
        "",
        f"> Framework: **RAGAS** · Test cases: **{len(comparison['hybrid']['rows'])}** · "
        "Configs: hybrid + RRF versus dense-only.",
        "",
        "## Overall scores",
        "",
        "| Metric | Hybrid + RRF | Dense-only | Delta |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
        "average": "Average",
    }
    for metric in (*METRICS, "average"):
        lines.append(
            f"| {labels[metric]} | {hybrid[metric]:.3f} | {dense[metric]:.3f} | "
            f"{hybrid[metric] - dense[metric]:+.3f} |"
        )

    ranked = sorted(
        comparison["hybrid"]["rows"],
        key=lambda row: mean(float(row.get(metric) or 0) for metric in METRICS),
    )[:3]
    lines += [
        "",
        "## Worst performers",
        "",
        "| Question | Faithfulness | Relevance | Recall | Precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        question = row["question"].replace("|", "\\|")
        values = [float(row.get(metric) or 0) for metric in METRICS]
        lines.append(f"| {question} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[3]:.3f} |")

    lines += [
        "",
        "## Analysis and recommendations",
        "",
        "- Inspect `evaluation_details.json` for each answer and its retrieved contexts before changing prompts.",
        "- If context recall is low, add comparison-query decomposition and keep adjacent band rows together when chunking.",
        "- If context precision is low, tune RRF/top-k or add a cross-encoder reranker.",
        "- If faithfulness is low while retrieval is strong, tighten the generation prompt and citation validation.",
        "",
        "> Scores are written by the evaluation script; they are not manually estimated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    DETAILS_PATH.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Use 18 for the required full evaluation")
    args = parser.parse_args()
    dataset = load_golden_dataset()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    dataset = dataset[: min(args.limit, len(dataset))]

    from src import task10_generation as pipeline

    comparison = compare_configs(pipeline, dataset)
    export_results(comparison)
    print(f"Saved {RESULTS_PATH} and {DETAILS_PATH}")


if __name__ == "__main__":
    main()
