from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chunking import Chunk, chunk_words

QASPER_REVISION = "cc58ffb39db7ff6ce1951e28e029996bf499304e"
QASPER_BASE_URL = (
    f"https://huggingface.co/datasets/allenai/qasper/resolve/{QASPER_REVISION}/qasper"
)
QASPER_PARQUET_FILES = {
    "train": f"{QASPER_BASE_URL}/qasper-train.parquet",
    "validation": f"{QASPER_BASE_URL}/qasper-validation.parquet",
    "test": f"{QASPER_BASE_URL}/qasper-test.parquet",
}


@dataclass(frozen=True)
class QAExample:
    doc_id: str
    question_id: str
    title: str
    question: str
    gold_answers: list[str]
    evidence: list[str]


def load_qasper(split: str = "validation"):
    if split not in QASPER_PARQUET_FILES:
        valid_splits = ", ".join(QASPER_PARQUET_FILES)
        raise ValueError(f"Unknown split '{split}'. Expected one of: {valid_splits}")
    from datasets import load_dataset

    return load_dataset("parquet", data_files={split: QASPER_PARQUET_FILES[split]}, split=split)


def document_text(record: dict[str, Any]) -> str:
    """Return the full paper text used for long-context length checks."""
    parts = []
    abstract = str(record.get("abstract", "")).strip()
    if abstract:
        parts.append(abstract)

    full_text = record.get("full_text", {})
    sections = full_text.get("section_name", [])
    paragraphs_by_section = full_text.get("paragraphs", [])
    for section, paragraphs in zip(sections, paragraphs_by_section):
        section_parts = [str(section).strip()] if str(section).strip() else []
        section_parts.extend(str(paragraph).strip() for paragraph in paragraphs if str(paragraph).strip())
        if section_parts:
            parts.append("\n".join(section_parts))
    return "\n\n".join(parts)


def document_word_count(record: dict[str, Any]) -> int:
    return len(document_text(record).split())


def is_long_context_record(record: dict[str, Any], *, min_words: int = 3000) -> bool:
    return document_word_count(record) >= min_words


def _normalise_answer(answer: dict[str, Any]) -> str | None:
    data = answer.get("answer", answer)
    if data.get("unanswerable"):
        return "Unanswerable"
    if data.get("free_form_answer"):
        return str(data["free_form_answer"]).strip()
    if data.get("extractive_spans"):
        spans = [str(span).strip() for span in data["extractive_spans"] if str(span).strip()]
        if spans:
            return " ; ".join(spans)
    yes_no = data.get("yes_no")
    if yes_no is not None:
        return str(yes_no)
    return None


def _normalise_evidence(answer: dict[str, Any]) -> list[str]:
    data = answer.get("answer", answer)
    evidence = data.get("evidence", answer.get("evidence", []))
    if not evidence:
        return []
    return [str(item).strip() for item in evidence if str(item).strip()]


def iter_answer_records(answers: Any) -> list[dict[str, Any]]:
    """Return answer records from Qasper's possible nested formats.

    The original HF script exposes answers as a list of records. The Parquet
    export can expose one question's answers as a dict of columns:
    `{"answer": [...], "annotation_id": [...], "worker_id": [...]}`.
    """
    if isinstance(answers, list):
        return [answer for answer in answers if isinstance(answer, dict)]
    if not isinstance(answers, dict):
        return []

    answer_values = answers.get("answer", [])
    annotation_ids = answers.get("annotation_id", [])
    worker_ids = answers.get("worker_id", [])

    if isinstance(answer_values, dict):
        answer_values = [answer_values]
    if not isinstance(answer_values, list):
        return []

    records = []
    for index, answer_value in enumerate(answer_values):
        record = {"answer": answer_value}
        if isinstance(annotation_ids, list) and index < len(annotation_ids):
            record["annotation_id"] = annotation_ids[index]
        if isinstance(worker_ids, list) and index < len(worker_ids):
            record["worker_id"] = worker_ids[index]
        records.append(record)
    return records


def extract_qa_examples(record: dict[str, Any]) -> list[QAExample]:
    qas = record.get("qas", {})
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answers_list = qas.get("answers", [])

    examples: list[QAExample] = []
    for question, question_id, answers in zip(questions, question_ids, answers_list):
        gold_answers = []
        evidence = []
        for answer in iter_answer_records(answers):
            normalised = _normalise_answer(answer)
            if normalised:
                gold_answers.append(normalised)
            evidence.extend(_normalise_evidence(answer))
        examples.append(
            QAExample(
                doc_id=record["id"],
                question_id=question_id,
                title=record.get("title", ""),
                question=question,
                gold_answers=gold_answers,
                evidence=evidence,
            )
        )
    return examples


def build_document_chunks(
    record: dict[str, Any],
    *,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    full_text = record.get("full_text", {})
    sections = full_text.get("section_name", [])
    paragraphs_by_section = full_text.get("paragraphs", [])
    chunks: list[Chunk] = []

    chunk_index = 0
    abstract = record.get("abstract", "")
    for text in chunk_words(abstract, chunk_size=chunk_size, overlap=overlap):
        chunks.append(
            Chunk(
                chunk_id=f"{record['id']}::abstract::{chunk_index}",
                doc_id=record["id"],
                title=record.get("title", ""),
                section="abstract",
                text=text,
            )
        )
        chunk_index += 1

    for section, paragraphs in zip(sections, paragraphs_by_section):
        section_text = " ".join(str(paragraph) for paragraph in paragraphs if str(paragraph).strip())
        for text in chunk_words(section_text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{record['id']}::{chunk_index}",
                    doc_id=record["id"],
                    title=record.get("title", ""),
                    section=str(section),
                    text=text,
                )
            )
            chunk_index += 1
    return chunks
