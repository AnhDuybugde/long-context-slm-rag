from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from .chunking import Chunk, chunk_words


@dataclass(frozen=True)
class QAExample:
    doc_id: str
    question_id: str
    title: str
    question: str
    gold_answers: list[str]
    evidence: list[str]


def load_qasper(split: str = "validation"):
    return load_dataset("allenai/qasper", split=split)


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


def extract_qa_examples(record: dict[str, Any]) -> list[QAExample]:
    qas = record.get("qas", {})
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answers_list = qas.get("answers", [])

    examples: list[QAExample] = []
    for question, question_id, answers in zip(questions, question_ids, answers_list):
        gold_answers = []
        evidence = []
        for answer in answers:
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
