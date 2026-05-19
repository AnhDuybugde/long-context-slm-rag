from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .chunking import Chunk
from .data import build_document_chunks
from .generator import SmallSeq2SeqGenerator
from .retriever import DenseRetriever


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.term_freqs = []
        self.doc_freqs = Counter()
        self.doc_lengths = []

        for chunk in chunks:
            terms = tokenize(chunk.text)
            term_freq = Counter(terms)
            self.term_freqs.append(term_freq)
            self.doc_lengths.append(len(terms))
            self.doc_freqs.update(term_freq.keys())

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_terms = tokenize(query)
        scores = []
        total_docs = len(self.chunks)

        for index, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_length = self.doc_lengths[index]
            for term in query_terms:
                if term not in term_freq:
                    continue
                doc_freq = self.doc_freqs[term]
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                frequency = term_freq[term]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append(score)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[index], float(scores[index])) for index in ranked_indices if scores[index] > 0]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[tuple[Chunk, float]]],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked_list, start=1):
            scores[chunk.chunk_id] += 1 / (rrf_k + rank)
            chunks_by_id[chunk.chunk_id] = chunk

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(chunks_by_id[chunk_id], scores[chunk_id]) for chunk_id in ranked_ids]


def u_shaped_reorder(chunks: list[Chunk]) -> list[Chunk]:
    """Place strongest chunks at the beginning and end of the context."""
    reordered: list[Chunk | None] = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1
    return [chunk for chunk in reordered if chunk is not None]


class ImprovedSeq2SeqGenerator(SmallSeq2SeqGenerator):
    def answer(
        self,
        question: str,
        contexts: list[Chunk],
        *,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 128,
    ) -> str:
        context_text = "\n\n".join(
            f"[{index + 1}] {chunk.section}\n{chunk.text}" for index, chunk in enumerate(contexts)
        )
        prompt = (
            "You are answering questions about a scientific paper. "
            "Use only the evidence in the context. "
            "Prefer short exact phrases from the context. "
            "If the answer is not supported, write Unanswerable.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context_text}\n\n"
            "Answer with one concise sentence or phrase:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class ImprovedRAGPipeline:
    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        retrieve_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.dense = DenseRetriever(retriever_model)
        self.sparse = BM25Retriever()
        self.generator = ImprovedSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def index_document(self, record: dict) -> None:
        chunks = build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap)
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def answer(self, question: str) -> dict:
        dense_results = self.dense.search(question, top_k=self.retrieve_k)
        sparse_results = self.sparse.search(question, top_k=self.retrieve_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=self.top_k)
        contexts = u_shaped_reorder([chunk for chunk, _score in fused])
        score_by_id = {chunk.chunk_id: score for chunk, score in fused}
        answer = self.generator.answer(question, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
            "scores": [score_by_id[chunk.chunk_id] for chunk in contexts],
        }

