from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from tqdm import tqdm

from .chunking import Chunk
from .data import QAExample, extract_qa_examples
from .metrics import (
    answer_relevancy,
    answer_string_recall,
    best_f1,
    context_precision,
    context_recall,
    faithfulness,
)
from .pipeline import BaseRAGPipeline


class RAGPipeline(Protocol):
    def index_document(self, record: dict[str, Any]) -> None:
        ...

    def answer(self, question: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class BaseRAGConfig:
    split: str = "validation"
    limit: int | None = None
    top_k: int = 5
    chunk_size: int = 180
    overlap: int = 40
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model: str = "google/flan-t5-base"
    output_predictions: str = "outputs/base_rag_qasper_predictions.jsonl"
    output_summary: str = "outputs/base_rag_qasper_summary.json"


@dataclass(frozen=True)
class EvaluationResult:
    doc_id: str
    question_id: str
    title: str
    question: str
    prediction: str
    gold_answers: list[str]
    evidence: list[str]
    token_f1: float
    answer_string_recall_at_k: float
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    contexts: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.docs = 0
        self.index_seconds = 0.0
        self.answer_seconds = 0.0
        self.token_f1 = 0.0
        self.answer_string_recall_at_k = 0.0
        self.context_precision = 0.0
        self.context_recall = 0.0
        self.faithfulness = 0.0
        self.answer_relevancy = 0.0
        self.generated = 0
        self.abstained = 0
        self.answered_token_f1 = 0.0
        self.abstained_has_gold_context = 0
        self.generated_without_gold_context = 0

    def add(self, result: EvaluationResult) -> None:
        self.count += 1
        self.token_f1 += result.token_f1
        self.answer_string_recall_at_k += result.answer_string_recall_at_k
        self.context_precision += result.context_precision
        self.context_recall += result.context_recall
        self.faithfulness += result.faithfulness
        self.answer_relevancy += result.answer_relevancy
        route = result.metadata.get("route")
        generated = route == "generate" or (route is None and result.prediction.strip().lower() != "unanswerable")
        has_gold_context = result.answer_string_recall_at_k > 0.0 or result.context_recall > 0.0
        if generated:
            self.generated += 1
            self.answered_token_f1 += result.token_f1
            if not has_gold_context:
                self.generated_without_gold_context += 1
        else:
            self.abstained += 1
            if has_gold_context:
                self.abstained_has_gold_context += 1

    def add_index_time(self, seconds: float) -> None:
        self.docs += 1
        self.index_seconds += seconds

    def add_answer_time(self, seconds: float) -> None:
        self.answer_seconds += seconds

    def summary(self, *, top_k: int) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "examples": 0,
                "docs_indexed": self.docs,
                "total_index_seconds": self.index_seconds,
                "avg_index_seconds_per_doc": self.index_seconds / self.docs if self.docs else 0.0,
                "total_answer_seconds": self.answer_seconds,
                "avg_answer_seconds_per_example": 0.0,
                "avg_token_f1": 0.0,
                f"avg_answer_string_recall_at_{top_k}": 0.0,
                "avg_context_precision": 0.0,
                "avg_context_recall": 0.0,
                "avg_faithfulness": 0.0,
                "avg_answer_relevancy": 0.0,
                "coverage": 0.0,
                "abstain_rate": 0.0,
                "answered_examples": 0,
                "abstained_examples": 0,
                "answered_token_f1": 0.0,
                "abstained_has_gold_context_rate": 0.0,
                "generated_without_gold_context_rate": 0.0,
            }
        return {
            "examples": self.count,
            "docs_indexed": self.docs,
            "total_index_seconds": self.index_seconds,
            "avg_index_seconds_per_doc": self.index_seconds / self.docs if self.docs else 0.0,
            "total_answer_seconds": self.answer_seconds,
            "avg_answer_seconds_per_example": self.answer_seconds / self.count,
            "avg_token_f1": self.token_f1 / self.count,
            f"avg_answer_string_recall_at_{top_k}": self.answer_string_recall_at_k / self.count,
            "avg_context_precision": self.context_precision / self.count,
            "avg_context_recall": self.context_recall / self.count,
            "avg_faithfulness": self.faithfulness / self.count,
            "avg_answer_relevancy": self.answer_relevancy / self.count,
            "coverage": self.generated / self.count,
            "abstain_rate": self.abstained / self.count,
            "answered_examples": self.generated,
            "abstained_examples": self.abstained,
            "answered_token_f1": self.answered_token_f1 / self.generated if self.generated else 0.0,
            "abstained_has_gold_context_rate": (
                self.abstained_has_gold_context / self.abstained if self.abstained else 0.0
            ),
            "generated_without_gold_context_rate": (
                self.generated_without_gold_context / self.generated if self.generated else 0.0
            ),
        }


class BaseRAGTrainer:
    """Run the baseline RAG experiment and persist comparable artifacts."""

    def __init__(
        self,
        config: BaseRAGConfig,
        *,
        pipeline: RAGPipeline | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline or BaseRAGPipeline(
            retriever_model=config.retriever_model,
            generator_model=config.generator_model,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            top_k=config.top_k,
        )

    def run(self, dataset: Iterable[dict[str, Any]]) -> dict[str, Any]:
        predictions_path = Path(self.config.output_predictions)
        summary_path = Path(self.config.output_summary)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = MetricsAccumulator()
        with predictions_path.open("w", encoding="utf-8") as file:
            for record in tqdm(dataset, desc="Running base RAG"):
                index_start = time.perf_counter()
                self.pipeline.index_document(record)
                metrics.add_index_time(time.perf_counter() - index_start)
                for example in extract_qa_examples(record):
                    answer_start = time.perf_counter()
                    result = self.evaluate_example(example)
                    answer_seconds = time.perf_counter() - answer_start
                    metrics.add_answer_time(answer_seconds)
                    result.metadata["answer_seconds"] = answer_seconds
                    metrics.add(result)
                    file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    if self.config.limit is not None and metrics.count >= self.config.limit:
                        return self._write_summary(metrics, summary_path, predictions_path)

        return self._write_summary(metrics, summary_path, predictions_path)

    def evaluate_example(self, example: QAExample) -> EvaluationResult:
        if hasattr(self.pipeline, "answer_example"):
            answer_result = self.pipeline.answer_example(example)
        else:
            answer_result = self.pipeline.answer(example.question)
        contexts = answer_result["contexts"]
        scores = answer_result["scores"]
        prediction = answer_result["answer"]
        metadata = {
            key: value
            for key, value in answer_result.items()
            if key not in {"answer", "contexts", "scores"}
        }

        return EvaluationResult(
            doc_id=example.doc_id,
            question_id=example.question_id,
            title=example.title,
            question=example.question,
            prediction=prediction,
            gold_answers=example.gold_answers,
            evidence=example.evidence,
            token_f1=best_f1(prediction, example.gold_answers),
            answer_string_recall_at_k=answer_string_recall(contexts, example.gold_answers),
            context_precision=context_precision(contexts, example.gold_answers, example.evidence),
            context_recall=context_recall(contexts, example.gold_answers, example.evidence),
            faithfulness=faithfulness(prediction, contexts),
            answer_relevancy=answer_relevancy(prediction, example.question, example.gold_answers),
            contexts=self._serialise_contexts(contexts, scores),
            metadata=metadata,
        )

    def _write_summary(
        self,
        metrics: MetricsAccumulator,
        summary_path: Path,
        predictions_path: Path,
    ) -> dict[str, Any]:
        summary = {
            "config": asdict(self.config),
            "metrics": metrics.summary(top_k=self.config.top_k),
            "predictions_path": str(predictions_path),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _serialise_contexts(contexts: list[Chunk], scores: list[float]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "text": chunk.text,
                "score": score,
            }
            for chunk, score in zip(contexts, scores)
        ]
