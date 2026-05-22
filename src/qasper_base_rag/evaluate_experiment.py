from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .data import is_long_context_record, load_qasper
from .experiment_pipelines import build_experiment_pipeline
from .trainer import BaseRAGConfig, BaseRAGTrainer

VARIANTS = [
    "base_dense",
    "bm25_only",
    "dense_u_shape",
    "dense_recency_heavy",
    "hybrid_rrf",
    "semantic_chunking_dense",
    "semantic_chunking_reranker",
    "semantic_chunking_hybrid_reranker",
    "dense_reranker",
    "raptor_extractive",
    "raptor_gmm_abstractive",
    "raptor_leiden_abstractive",
    "raptor_agglomerative_abstractive",
    "semantic_raptor_leiden_reranker",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one independent Qasper long-context RAG experiment.")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of QA examples to evaluate. Omit to run the full selected split.",
    )
    parser.add_argument(
        "--min-doc-words",
        type=int,
        default=3000,
        help="Only evaluate papers with at least this many words. Use 0 to disable long-context filtering.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--generator-model", default="google/flan-t5-base")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-dir", default="outputs/experiments")
    return parser.parse_args()


def long_context_records(dataset: Iterable[dict], *, min_doc_words: int) -> Iterable[dict]:
    for record in dataset:
        if min_doc_words <= 0 or is_long_context_record(record, min_words=min_doc_words):
            yield record


def main() -> None:
    args = parse_args()
    dataset = load_qasper(args.split)
    selected_dataset = long_context_records(dataset, min_doc_words=args.min_doc_words)

    output_stem = f"{args.variant}_{args.split}_min{args.min_doc_words}"
    config = BaseRAGConfig(
        split=args.split,
        limit=args.limit,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        retriever_model=args.retriever_model,
        generator_model=args.generator_model,
        output_predictions=f"{args.output_dir}/{output_stem}_predictions.jsonl",
        output_summary=f"{args.output_dir}/{output_stem}_summary.json",
    )
    pipeline = build_experiment_pipeline(
        args.variant,
        retriever_model=args.retriever_model,
        generator_model=args.generator_model,
        reranker_model=args.reranker_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        retrieve_k=args.retrieve_k,
        top_k=args.top_k,
    )
    summary = BaseRAGTrainer(config, pipeline=pipeline).run(selected_dataset)
    summary["variant"] = args.variant
    summary["long_context_filter"] = {"min_doc_words": args.min_doc_words}
    Path(config.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"variant={args.variant}")
    print(f"long_context_min_doc_words={args.min_doc_words}")
    for key, value in summary["metrics"].items():
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")
    print(f"predictions={summary['predictions_path']}")
    print(f"summary={config.output_summary}")
    print(f"config={asdict(config)}")


if __name__ == "__main__":
    main()
