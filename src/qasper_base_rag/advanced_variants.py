from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_words
from .data import build_document_chunks
from .generator import SmallSeq2SeqGenerator
from .improved import u_shaped_reorder
from .metrics import token_overlap_recall
from .retriever import DenseRetriever


def split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _encode_texts(embedder: Any, texts: list[str]) -> np.ndarray:
    try:
        vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except TypeError:
        vectors = embedder.encode(texts)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)

@dataclass(frozen=True)
class SemanticChunkingConfig:
    min_words: int = 60
    max_words: int = 220
    breakpoint_threshold: float = 0.35
    overlap_sentences: int = 1


class SemanticChunker:
    """Sentence-boundary chunker that breaks on cosine-distance jumps."""

    def __init__(self, config: SemanticChunkingConfig | None = None, *, embedder: Any | None = None) -> None:
        self.config = config or SemanticChunkingConfig()
        self.embedder = embedder

    def chunk_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if self.embedder is None or len(sentences) == 1:
            return chunk_words(text, chunk_size=self.config.max_words, overlap=0)

        try:
            embeddings = _encode_texts(self.embedder, sentences)
        except Exception:
            return chunk_words(text, chunk_size=self.config.max_words, overlap=0)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_words = len(sentences[0].split())

        for index in range(1, len(sentences)):
            sentence = sentences[index]
            sentence_words = len(sentence.split())
            similarity = float(np.dot(embeddings[index - 1], embeddings[index]))
            distance = 1.0 - similarity
            too_large = current_words + sentence_words > self.config.max_words
            semantic_break = current_words >= self.config.min_words and distance >= self.config.breakpoint_threshold

            if too_large or semantic_break:
                chunks.append(" ".join(current).strip())
                overlap = current[-self.config.overlap_sentences :] if self.config.overlap_sentences else []
                current = [*overlap, sentence]
                current_words = sum(len(item.split()) for item in current)
            else:
                current.append(sentence)
                current_words += sentence_words

        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]


def build_semantic_document_chunks(
    record: dict[str, Any],
    *,
    chunker: SemanticChunker,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    abstract = str(record.get("abstract", "")).strip()
    for text in chunker.chunk_text(abstract):
        chunks.append(Chunk(f"{record['id']}::semantic::abstract::{chunk_index}", record["id"], record.get("title", ""), "abstract", text))
        chunk_index += 1

    full_text = record.get("full_text", {})
    sections = full_text.get("section_name", [])
    paragraphs_by_section = full_text.get("paragraphs", [])
    for section, paragraphs in zip(sections, paragraphs_by_section):
        section_text = " ".join(str(paragraph) for paragraph in paragraphs if str(paragraph).strip())
        for text in chunker.chunk_text(section_text):
            chunks.append(Chunk(f"{record['id']}::semantic::{chunk_index}", record["id"], record.get("title", ""), str(section), text))
            chunk_index += 1
    return chunks


class SemanticDensePipeline:
    """Dense retrieval with semantic sentence-boundary chunking."""

    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        min_words: int = 60,
        max_words: int = 220,
        breakpoint_threshold: float = 0.35,
        top_k: int = 5,
    ) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(
                min_words=min_words,
                max_words=max_words,
                breakpoint_threshold=breakpoint_threshold,
            ),
            embedder=self.retriever.model,
        )
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        self.retriever.index(build_semantic_document_chunks(record, chunker=self.chunker))

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [chunk for chunk, _score in retrieved]
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score for _chunk, score in retrieved],
        }


class CrossEncoderReranker:
    """Cross-encoder reranker with a lexical fallback for offline tests."""

    def __init__(self, model_name: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.load_error: str | None = None
        if model_name is None:
            self.load_error = "disabled"
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception as error:
            self.load_error = str(error)

    def rerank(
        self,
        question: str,
        candidates: list[tuple[Chunk, float]],
        *,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        if self.model is not None:
            pairs = [(question, chunk.text) for chunk, _score in candidates]
            scores = [float(score) for score in self.model.predict(pairs)]
        else:
            scores = [self._lexical_fallback_score(question, chunk, score) for chunk, score in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [(chunk, score) for ((chunk, _original_score), score) in ranked]

    @staticmethod
    def _lexical_fallback_score(question: str, chunk: Chunk, original_score: float) -> float:
        return float(original_score) + token_overlap_recall(chunk.text, question)


class DenseRerankerPipeline:
    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 180,
        overlap: int = 40,
        retrieve_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        self.retriever.index(build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap))

    def answer(self, question: str) -> dict[str, Any]:
        candidates = self.retriever.search(question, top_k=self.retrieve_k)
        reranked = self.reranker.rerank(question, candidates, top_k=self.top_k)
        contexts = [chunk for chunk, _score in reranked]
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score for _chunk, score in reranked],
            "reranker_model": self.reranker.model_name,
            "reranker_load_error": self.reranker.load_error,
        }

@dataclass(frozen=True)
class RaptorConfig:
    group_size: int = 4
    max_levels: int = 2
    max_summary_words: int = 90
    similarity_threshold: float = 0.45
    nearest_neighbors: int = 8
    use_leiden: bool = True


class AbstractiveClusterSummarizer:
    """Summarize RAPTOR clusters with the existing seq2seq generator model."""

    def __init__(
        self,
        generator: SmallSeq2SeqGenerator,
        *,
        max_input_tokens: int = 768,
        max_new_tokens: int = 96,
    ) -> None:
        self.generator = generator
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def summarize(self, chunks: list[Chunk]) -> str:
        source_text = "\n\n".join(f"{chunk.section}: {chunk.text}" for chunk in chunks)
        prompt = (
            "Summarize the shared evidence from these scientific-paper passages. "
            "Keep named methods, datasets, metrics, and conclusions. "
            "Do not add information not present in the passages.\n\n"
            f"Passages:\n{source_text}\n\nSummary:"
        )
        inputs = self.generator.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.generator.device)
        outputs = self.generator.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=2,
        )
        return self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class RaptorTreeBuilder:
    """Lightweight RAPTOR-style collapsed tree using extractive parent summaries."""

    def __init__(
        self,
        config: RaptorConfig | None = None,
        *,
        embedder: Any | None = None,
        summarizer: Any | None = None,
    ) -> None:
        self.config = config or RaptorConfig()
        self.embedder = embedder
        self.summarizer = summarizer
        self.last_backend = "not_run"

    def build(self, leaves: list[Chunk]) -> list[Chunk]:
        parents: list[Chunk] = []
        current = leaves
        for level in range(1, self.config.max_levels + 1):
            if len(current) <= 1:
                break
            groups = self._groups(current)
            level_parents = []
            for group_index, group in enumerate(groups):
                if len(group) <= 1:
                    continue
                summary = self._summarize(group)
                first = group[0]
                parent = Chunk(
                    chunk_id=f"{first.doc_id}::raptor::level{level}::{group_index}",
                    doc_id=first.doc_id,
                    title=first.title,
                    section=f"raptor_level_{level}",
                    text=summary,
                )
                level_parents.append(parent)
            if not level_parents:
                break
            parents.extend(level_parents)
            current = level_parents
        return parents

    def _groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if self.embedder is None or len(chunks) <= self.config.group_size:
            self.last_backend = "contiguous"
            return self._contiguous_groups(chunks)
        try:
            embeddings = _encode_texts(self.embedder, [chunk.text for chunk in chunks])
        except Exception:
            self.last_backend = "embedding_failed_contiguous"
            return self._contiguous_groups(chunks)

        if self.config.use_leiden:
            try:
                return self._leiden_groups(chunks, embeddings)
            except Exception:
                pass

        self.last_backend = "similarity_components"
        visited: set[int] = set()
        groups: list[list[int]] = []
        for index in range(len(chunks)):
            if index in visited:
                continue
            stack = [index]
            component = []
            visited.add(index)
            while stack:
                current = stack.pop()
                component.append(current)
                similarities = embeddings @ embeddings[current]
                neighbours = np.where(similarities >= self.config.similarity_threshold)[0]
                for neighbour in neighbours.tolist():
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
            groups.extend(self._split_indices(sorted(component)))
        return [[chunks[index] for index in group] for group in groups]

    def _leiden_groups(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[list[Chunk]]:
        import igraph as ig
        import leidenalg

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        similarities = embeddings @ embeddings.T
        for index in range(len(chunks)):
            neighbour_indices = np.argsort(similarities[index])[::-1]
            added = 0
            for neighbour in neighbour_indices:
                if neighbour == index:
                    continue
                score = float(similarities[index, neighbour])
                if score < self.config.similarity_threshold and added >= 1:
                    continue
                edge = (min(index, int(neighbour)), max(index, int(neighbour)))
                if edge not in edges:
                    edges.append(edge)
                    weights.append(max(score, 0.0))
                added += 1
                if added >= self.config.nearest_neighbors:
                    break

        if not edges:
            self.last_backend = "leiden_no_edges_contiguous"
            return self._contiguous_groups(chunks)

        graph = ig.Graph(n=len(chunks), edges=edges, directed=False)
        graph.es["weight"] = weights
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=graph.es["weight"],
        )
        groups: list[list[int]] = []
        for community in partition:
            community_indices = sorted(int(index) for index in community)
            groups.extend(self._split_indices(community_indices))
        self.last_backend = "leiden"
        return [[chunks[index] for index in group] for group in groups if group]

    def _contiguous_groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[index : index + self.config.group_size] for index in range(0, len(chunks), self.config.group_size)]

    def _split_indices(self, indices: list[int]) -> list[list[int]]:
        return [indices[index : index + self.config.group_size] for index in range(0, len(indices), self.config.group_size)]

    def _summarize(self, chunks: list[Chunk]) -> str:
        if self.summarizer is not None:
            summary = self.summarizer.summarize(chunks).strip()
            if summary:
                return summary
        summary_sentences = []
        for chunk in chunks:
            sentences = split_sentences(chunk.text)
            if sentences:
                summary_sentences.append(sentences[0])
        words = " ".join(summary_sentences).split()[: self.config.max_summary_words]
        return " ".join(words)


class RaptorExtractivePipeline:
    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        group_size: int = 4,
        top_k: int = 5,
    ) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.tree_builder = RaptorTreeBuilder(RaptorConfig(group_size=group_size), embedder=self.retriever.model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        leaves = build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap)
        parents = self.tree_builder.build(leaves)
        self.retriever.index([*leaves, *parents])

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = u_shaped_reorder([chunk for chunk, _score in retrieved])
        score_by_id = {chunk.chunk_id: score for chunk, score in retrieved}
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score_by_id[chunk.chunk_id] for chunk in contexts],
        }


class RaptorLeidenAbstractivePipeline:
    """More faithful RAPTOR: recursive summaries plus Leiden when available."""

    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        group_size: int = 4,
        top_k: int = 5,
    ) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.summarizer = AbstractiveClusterSummarizer(self.generator)
        self.tree_builder = RaptorTreeBuilder(
            RaptorConfig(group_size=group_size, use_leiden=True),
            embedder=self.retriever.model,
            summarizer=self.summarizer,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.parent_count = 0
        self.raptor_backend = "not_run"

    def index_document(self, record: dict[str, Any]) -> None:
        leaves = build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap)
        parents = self.tree_builder.build(leaves)
        self.parent_count = len(parents)
        self.raptor_backend = self.tree_builder.last_backend
        self.retriever.index([*leaves, *parents])

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = u_shaped_reorder([chunk for chunk, _score in retrieved])
        score_by_id = {chunk.chunk_id: score for chunk, score in retrieved}
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score_by_id[chunk.chunk_id] for chunk in contexts],
            "raptor_parent_count": self.parent_count,
            "raptor_backend": self.raptor_backend,
        }
