"""Run RAGAS evaluation for prompt versions v1 and v2.

Usage:
    python eval/run_eval.py

Prerequisites:
    1. Add OPENAI_API_KEY to .env.
    2. Build the Chroma vector store with `python ingest.py`.

Outputs:
    eval/results/ragas_per_question_scores.csv
    eval/results/ragas_summary_scores.csv
    eval/results/ragas_prompt_comparison.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from chain import build_chain_with_sources  # noqa: E402

SUPPORTED_PROMPT_VERSIONS = ("v1", "v2", "v3", "v4", "v5")
DEFAULT_PROMPT_VERSIONS = SUPPORTED_PROMPT_VERSIONS
DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("test_questions.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")
METADATA_COLUMNS = {
    "id",
    "category",
    "prompt_version",
    "question",
    "answer",
    "contexts",
    "ground_truth",
    "source_files",
}
OUTPUT_METADATA_COLUMNS = [
    "id",
    "category",
    "prompt_version",
    "question",
    "answer",
    "ground_truth",
    "source_files",
]


def clean_csv_text(value: Any) -> Any:
    """Collapse whitespace in long text fields so CSV rows stay one line."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def _load_metrics() -> list[Any]:
    """Import the standard RAGAS metrics across minor API differences."""
    try:
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return [faithfulness, answer_relevancy, context_precision, context_recall]
    except ImportError:
        from ragas.metrics import (  # type: ignore
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        return [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]


def _build_evaluator_models() -> dict[str, Any]:
    """Use the project OpenAI settings for RAGAS judge and embedding calls."""
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0,
        api_key=config.OPENAI_API_KEY,
    )
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        return {
            "llm": LangchainLLMWrapper(llm),
            "embeddings": LangchainEmbeddingsWrapper(embeddings),
        }
    except ImportError:
        return {"llm": llm, "embeddings": embeddings}


def load_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or len(questions) < 10:
        raise ValueError("test_questions.json must contain a list of at least 10 questions.")

    required_fields = {"id", "category", "question", "ground_truth"}
    for item in questions:
        missing = required_fields - set(item)
        if missing:
            raise ValueError(f"Question item is missing required fields: {sorted(missing)}")

    return questions


def strip_internal_thinking(answer: str) -> str:
    """Remove any leaked v2 thinking block before customer-facing evaluation."""
    cleaned = re.sub(
        r"<thinking>.*?</thinking>\s*",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def generate_answers(
    questions: list[dict[str, str]],
    prompt_versions: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for prompt_version in prompt_versions:
        print(f"Running {prompt_version} against {len(questions)} questions...")
        chain = build_chain_with_sources(prompt_version=prompt_version)

        for i, item in enumerate(questions, start=1):
            question = item["question"]
            print(f"  [{i:02d}/{len(questions)}] {item['id']}: {question}")
            result = chain.invoke(question)
            docs = result["context_docs"]
            answer = strip_internal_thinking(result["answer"])

            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt_version": prompt_version,
                    "question": question,
                    "answer": answer,
                    "contexts": [doc.page_content for doc in docs],
                    "ground_truth": item["ground_truth"],
                    "source_files": "; ".join(
                        str(doc.metadata.get("source", "unknown")) for doc in docs
                    ),
                }
            )

    return rows


def evaluate_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Run RAGAS and return one scored row per generated answer."""
    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )

    result = evaluate(
        dataset,
        metrics=_load_metrics(),
        **_build_evaluator_models(),
    )
    scores = result.to_pandas()

    metadata = pd.DataFrame(rows)[OUTPUT_METADATA_COLUMNS].map(clean_csv_text)
    metric_columns = [
        column
        for column in scores.columns
        if pd.api.types.is_numeric_dtype(scores[column])
    ]
    scores = pd.concat(
        [
            metadata.reset_index(drop=True),
            scores[metric_columns].reset_index(drop=True),
        ],
        axis=1,
    )

    return scores


def summarise_scores(per_question_scores: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in per_question_scores.columns
        if column not in OUTPUT_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(per_question_scores[column])
    ]

    if not metric_columns:
        raise ValueError("RAGAS did not return any numeric metric columns to summarise.")

    return (
        per_question_scores.groupby("prompt_version", as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .round(4)
    )


def write_comparison_chart(summary_scores: pd.DataFrame, output_path: Path) -> None:
    chart_data = summary_scores.set_index("prompt_version").T
    ax = chart_data.plot(kind="bar", figsize=(12, 6), rot=30)
    ax.set_title("RAGAS Comparison Across Prompt Versions")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Average score")
    ax.set_ylim(0, 1)
    ax.legend(title="Prompt version")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG prompt versions with RAGAS.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to test_questions.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and chart outputs.",
    )
    parser.add_argument(
        "--prompt-versions",
        nargs="+",
        default=DEFAULT_PROMPT_VERSIONS,
        choices=SUPPORTED_PROMPT_VERSIONS,
        help="Prompt versions to evaluate. Defaults to v1 v2 v3 v4 v5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.require_api_key()

    questions = load_questions(args.questions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_answers(questions, tuple(args.prompt_versions))
    per_question_scores = evaluate_rows(rows)
    summary_scores = summarise_scores(per_question_scores)

    per_question_path = output_dir / "ragas_per_question_scores.csv"
    summary_path = output_dir / "ragas_summary_scores.csv"
    chart_path = output_dir / "ragas_prompt_comparison.png"

    per_question_scores.to_csv(per_question_path, index=False)
    summary_scores.to_csv(summary_path, index=False)
    write_comparison_chart(summary_scores, chart_path)

    print("\nEvaluation complete.")
    print(f"Per-question scores: {per_question_path}")
    print(f"Summary scores     : {summary_path}")
    print(f"Comparison chart   : {chart_path}")
    print("\nSummary:")
    print(summary_scores.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
