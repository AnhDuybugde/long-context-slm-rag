from __future__ import annotations

import argparse

from .data import load_qasper
from .trainer import BaseRAGConfig, BaseRAGTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the base Qasper RAG pipeline.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of QA examples to evaluate. Omit to run the full split.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--generator-model", default="google/flan-t5-base")
    parser.add_argument("--output-predictions", default="outputs/base_rag_qasper_predictions.jsonl")
    parser.add_argument("--output-summary", default="outputs/base_rag_qasper_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_qasper(args.split)
    config = BaseRAGConfig(
        split=args.split,
        limit=args.limit,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        retriever_model=args.retriever_model,
        generator_model=args.generator_model,
        output_predictions=args.output_predictions,
        output_summary=args.output_summary,
    )
    summary = BaseRAGTrainer(config).run(dataset)
    for key, value in summary["metrics"].items():
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")
    print(f"predictions={summary['predictions_path']}")
    print(f"summary={args.output_summary}")


if __name__ == "__main__":
    main()
