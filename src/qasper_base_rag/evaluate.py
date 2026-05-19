from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .data import extract_qa_examples, load_qasper
from .metrics import (
    answer_relevancy,
    answer_string_recall,
    best_f1,
    context_precision,
    context_recall,
    faithfulness,
)
from .pipeline import BaseRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the base Qasper RAG pipeline.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=20, help="Number of QA examples to evaluate.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--generator-model", default="google/flan-t5-base")
    parser.add_argument("--output", default="outputs/base_rag_qasper_predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_qasper(args.split)
    pipeline = BaseRAGPipeline(
        retriever_model=args.retriever_model,
        generator_model=args.generator_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    f1_sum = 0.0
    retrieval_recall_sum = 0.0
    context_precision_sum = 0.0
    context_recall_sum = 0.0
    faithfulness_sum = 0.0
    answer_relevancy_sum = 0.0

    with output_path.open("w", encoding="utf-8") as file:
        for record in tqdm(dataset, desc="Evaluating Qasper documents"):
            pipeline.index_document(record)
            for example in extract_qa_examples(record):
                result = pipeline.answer(example.question)
                f1 = best_f1(result["answer"], example.gold_answers)
                retrieval_recall = answer_string_recall(result["contexts"], example.gold_answers)
                ctx_precision = context_precision(
                    result["contexts"],
                    example.gold_answers,
                    example.evidence,
                )
                ctx_recall = context_recall(
                    result["contexts"],
                    example.gold_answers,
                    example.evidence,
                )
                groundedness = faithfulness(result["answer"], result["contexts"])
                relevance = answer_relevancy(
                    result["answer"],
                    example.question,
                    example.gold_answers,
                )
                f1_sum += f1
                retrieval_recall_sum += retrieval_recall
                context_precision_sum += ctx_precision
                context_recall_sum += ctx_recall
                faithfulness_sum += groundedness
                answer_relevancy_sum += relevance
                total += 1

                file.write(
                    json.dumps(
                        {
                            "doc_id": example.doc_id,
                            "question_id": example.question_id,
                            "title": example.title,
                            "question": example.question,
                            "prediction": result["answer"],
                            "gold_answers": example.gold_answers,
                            "evidence": example.evidence,
                            "token_f1": f1,
                            "answer_string_recall_at_k": retrieval_recall,
                            "context_precision": ctx_precision,
                            "context_recall": ctx_recall,
                            "faithfulness": groundedness,
                            "answer_relevancy": relevance,
                            "contexts": [
                                {
                                    "chunk_id": chunk.chunk_id,
                                    "section": chunk.section,
                                    "text": chunk.text,
                                    "score": score,
                                }
                                for chunk, score in zip(result["contexts"], result["scores"])
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if total >= args.limit:
                    avg_f1 = f1_sum / total
                    avg_recall = retrieval_recall_sum / total
                    avg_context_precision = context_precision_sum / total
                    avg_context_recall = context_recall_sum / total
                    avg_faithfulness = faithfulness_sum / total
                    avg_answer_relevancy = answer_relevancy_sum / total
                    print(f"examples={total}")
                    print(f"avg_token_f1={avg_f1:.4f}")
                    print(f"avg_answer_string_recall_at_{args.top_k}={avg_recall:.4f}")
                    print(f"avg_context_precision={avg_context_precision:.4f}")
                    print(f"avg_context_recall={avg_context_recall:.4f}")
                    print(f"avg_faithfulness={avg_faithfulness:.4f}")
                    print(f"avg_answer_relevancy={avg_answer_relevancy:.4f}")
                    print(f"predictions={output_path}")
                    return


if __name__ == "__main__":
    main()
