from __future__ import annotations

import re
from collections import Counter

from .chunking import Chunk


def normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalise_text(prediction).split()
    gold_tokens = normalise_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1(prediction, gold) for gold in gold_answers)


def answer_string_recall(contexts: list[Chunk], gold_answers: list[str]) -> float:
    """Weak baseline retrieval check: does any gold answer string appear in context?"""
    if not gold_answers:
        return 0.0
    context_text = normalise_text(" ".join(chunk.text for chunk in contexts))
    hits = 0
    for answer in gold_answers:
        answer_text = normalise_text(answer)
        if answer_text and answer_text in context_text:
            hits += 1
    return hits / len(gold_answers)


def _token_set(text: str) -> set[str]:
    return set(normalise_text(text).split())


def token_overlap_recall(candidate: str, reference: str) -> float:
    candidate_tokens = _token_set(candidate)
    reference_tokens = _token_set(reference)
    if not reference_tokens:
        return 0.0
    return len(candidate_tokens & reference_tokens) / len(reference_tokens)


def context_recall(
    contexts: list[Chunk],
    gold_answers: list[str],
    evidence: list[str],
    *,
    overlap_threshold: float = 0.45,
) -> float:
    """Estimate whether retrieved context covers required evidence.

    Qasper contains evidence annotations for many answers. When evidence is
    unavailable, this falls back to checking gold answer strings in context.
    """
    references = evidence or gold_answers
    if not references:
        return 0.0
    context_text = " ".join(chunk.text for chunk in contexts)
    covered = 0
    for reference in references:
        if token_overlap_recall(context_text, reference) >= overlap_threshold:
            covered += 1
    return covered / len(references)


def context_precision(
    contexts: list[Chunk],
    gold_answers: list[str],
    evidence: list[str],
    *,
    overlap_threshold: float = 0.25,
) -> float:
    """Estimate the share of retrieved chunks that are useful evidence."""
    references = evidence or gold_answers
    if not contexts or not references:
        return 0.0
    relevant = 0
    for chunk in contexts:
        best_overlap = max(token_overlap_recall(chunk.text, reference) for reference in references)
        if best_overlap >= overlap_threshold:
            relevant += 1
    return relevant / len(contexts)


def answer_relevancy(prediction: str, question: str, gold_answers: list[str]) -> float:
    """Heuristic answer relevance score for baseline comparison.

    This is not an LLM judge. It combines similarity to the question and, when
    available, similarity to gold answers so the baseline can be compared later.
    """
    if not prediction:
        return 0.0
    question_similarity = token_f1(prediction, question)
    if gold_answers:
        gold_similarity = best_f1(prediction, gold_answers)
        return 0.35 * question_similarity + 0.65 * gold_similarity
    return question_similarity


def faithfulness(
    prediction: str,
    contexts: list[Chunk],
    *,
    overlap_threshold: float = 0.35,
) -> float:
    """Estimate whether answer sentences are supported by retrieved context."""
    if not prediction:
        return 0.0
    if normalise_text(prediction) == "unanswerable":
        return 1.0
    context_text = " ".join(chunk.text for chunk in contexts)
    claims = [
        claim.strip()
        for claim in re.split(r"[.!?;\n]+", prediction)
        if claim.strip()
    ]
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        if token_overlap_recall(context_text, claim) >= overlap_threshold:
            supported += 1
    return supported / len(claims)
