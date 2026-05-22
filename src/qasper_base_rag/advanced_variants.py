from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_words
from .data import build_document_chunks
from .generator import SmallSeq2SeqGenerator
from .improved import BM25Retriever, reciprocal_rank_fusion, u_shaped_reorder
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


class SemanticRerankerPipeline:
    """Semantic chunking followed by dense retrieval and cross-encoder reranking."""

    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_words: int = 60,
        max_words: int = 220,
        breakpoint_threshold: float = 0.35,
        retrieve_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(
                min_words=min_words,
                max_words=max_words,
                breakpoint_threshold=breakpoint_threshold,
            ),
            embedder=self.retriever.model,
        )
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        self.retriever.index(build_semantic_document_chunks(record, chunker=self.chunker))

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


class SemanticHybridRerankerPipeline:
    """Semantic chunking, dense+BM25 RRF candidate retrieval, then cross-encoder reranking."""

    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_words: int = 60,
        max_words: int = 220,
        breakpoint_threshold: float = 0.35,
        retrieve_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.dense = DenseRetriever(retriever_model)
        self.sparse = BM25Retriever()
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(
                min_words=min_words,
                max_words=max_words,
                breakpoint_threshold=breakpoint_threshold,
            ),
            embedder=self.dense.model,
        )
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        chunks = build_semantic_document_chunks(record, chunker=self.chunker)
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def answer(self, question: str) -> dict[str, Any]:
        dense_results = self.dense.search(question, top_k=self.retrieve_k)
        sparse_results = self.sparse.search(question, top_k=self.retrieve_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=self.retrieve_k)
        reranked = self.reranker.rerank(question, fused, top_k=self.top_k)
        contexts = [chunk for chunk, _score in reranked]
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score for _chunk, score in reranked],
            "reranker_model": self.reranker.model_name,
            "reranker_load_error": self.reranker.load_error,
            "candidate_retrieval": "semantic_dense_bm25_rrf",
        }

@dataclass(frozen=True)
class RaptorConfig:
    group_size: int = 4
    max_levels: int = 2
    max_summary_words: int = 90
    max_cluster_words: int = 3500
    similarity_threshold: float = 0.45
    adaptive_threshold: bool = False
    similarity_quantile: float = 0.75
    semantic_similarity_threshold: float = 0.70
    nearest_neighbors: int = 8
    neighbor_step: int = 2
    resolution: float = 1.0
    resolution_decay: float = 0.2
    min_resolution: float = 0.1
    use_leiden: bool = True
    random_state: int = 13
    gmm_threshold: float = 0.10
    reduction_dimension: int = 10
    max_gmm_clusters: int = 50
    position_priority: float = 5.0


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
                threshold = self._similarity_threshold(embeddings)
                neighbours = np.where(similarities >= threshold)[0]
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
        layer = self._layer_from_chunk_ids(chunks)
        threshold = self._similarity_threshold(embeddings)
        neighbour_limit = min(len(chunks) - 1, self.config.nearest_neighbors + layer * self.config.neighbor_step)
        resolution = max(self.config.min_resolution, self.config.resolution - layer * self.config.resolution_decay)
        for index in range(len(chunks)):
            neighbour_indices = np.argsort(similarities[index])[::-1]
            added = 0
            for neighbour in neighbour_indices:
                if neighbour == index:
                    continue
                score = float(similarities[index, neighbour])
                if score < threshold and added >= 1:
                    continue
                edge = (min(index, int(neighbour)), max(index, int(neighbour)))
                if edge not in edges:
                    edges.append(edge)
                    weights.append(max(score, 0.0))
                added += 1
                if added >= neighbour_limit:
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
            resolution_parameter=resolution,
            seed=self.config.random_state,
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

    def _similarity_threshold(self, embeddings: np.ndarray) -> float:
        if not self.config.adaptive_threshold or len(embeddings) <= 2:
            return self.config.similarity_threshold
        similarities = embeddings @ embeddings.T
        upper = similarities[np.triu_indices_from(similarities, k=1)]
        if upper.size == 0:
            return self.config.similarity_threshold
        threshold = float(np.quantile(upper, self.config.similarity_quantile))
        return max(min(threshold, 0.95), 0.10)

    @staticmethod
    def _layer_from_chunk_ids(chunks: list[Chunk]) -> int:
        for chunk in chunks:
            match = re.search(r"raptor::level(\d+)", chunk.chunk_id)
            if match:
                return int(match.group(1))
        return 0

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


class GMMRaptorTreeBuilder(RaptorTreeBuilder):
    """Original RAPTOR-style UMAP + soft GMM clustering with BIC cluster selection."""

    def _groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if self.embedder is None or len(chunks) <= self.config.group_size:
            self.last_backend = "contiguous"
            return self._contiguous_groups(chunks)
        try:
            embeddings = _encode_texts(self.embedder, [chunk.text for chunk in chunks])
        except Exception:
            self.last_backend = "gmm_failed_contiguous"
            return self._contiguous_groups(chunks)

        groups = self._raptor_clusters(chunks, embeddings)
        self.last_backend = "umap_gmm_soft" if groups else "gmm_empty_contiguous"
        return groups if groups else self._contiguous_groups(chunks)

    def _raptor_clusters(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[list[Chunk]]:
        global_embeddings = self._reduce_embeddings(
            embeddings,
            n_components=min(self.config.reduction_dimension, max(1, len(chunks) - 2)),
            n_neighbors=max(2, int((len(chunks) - 1) ** 0.5)),
        )
        global_labels, global_count = self._soft_gmm_labels(global_embeddings)

        grouped: list[list[Chunk]] = []
        for global_label in range(global_count):
            global_indices = [index for index, labels in enumerate(global_labels) if global_label in labels]
            if not global_indices:
                continue
            if len(global_indices) <= self.config.reduction_dimension + 1:
                local_groups = [global_indices]
            else:
                local_embeddings = self._reduce_embeddings(
                    embeddings[global_indices],
                    n_components=min(self.config.reduction_dimension, max(1, len(global_indices) - 2)),
                    n_neighbors=min(10, max(2, len(global_indices) - 1)),
                )
                local_labels, local_count = self._soft_gmm_labels(local_embeddings)
                local_groups = [
                    [global_indices[index] for index, labels in enumerate(local_labels) if local_label in labels]
                    for local_label in range(local_count)
                ]

            for indices in local_groups:
                cluster = [chunks[index] for index in indices]
                if not cluster:
                    continue
                if self._cluster_word_count(cluster) > self.config.max_cluster_words and len(cluster) > 1:
                    child_embeddings = embeddings[indices]
                    grouped.extend(self._raptor_clusters(cluster, child_embeddings))
                else:
                    grouped.append(cluster)
        return grouped

    def _reduce_embeddings(self, embeddings: np.ndarray, *, n_components: int, n_neighbors: int) -> np.ndarray:
        if len(embeddings) <= n_components + 1:
            return embeddings
        try:
            import umap

            return umap.UMAP(
                n_neighbors=min(n_neighbors, len(embeddings) - 1),
                n_components=n_components,
                metric="cosine",
                random_state=self.config.random_state,
            ).fit_transform(embeddings)
        except Exception:
            return embeddings

    def _soft_gmm_labels(self, embeddings: np.ndarray) -> tuple[list[np.ndarray], int]:
        from sklearn.mixture import GaussianMixture

        max_clusters = min(self.config.max_gmm_clusters, len(embeddings))
        if max_clusters <= 1:
            return [np.array([0]) for _ in range(len(embeddings))], 1

        cluster_range = np.arange(1, max_clusters)
        if len(cluster_range) == 0:
            return [np.array([0]) for _ in range(len(embeddings))], 1
        bics = []
        for cluster_count in cluster_range:
            model = GaussianMixture(n_components=int(cluster_count), random_state=self.config.random_state)
            model.fit(embeddings)
            bics.append(model.bic(embeddings))
        optimal_count = int(cluster_range[int(np.argmin(bics))])
        model = GaussianMixture(n_components=optimal_count, random_state=self.config.random_state)
        model.fit(embeddings)
        probabilities = model.predict_proba(embeddings)
        labels = [np.where(probability > self.config.gmm_threshold)[0] for probability in probabilities]
        labels = [label if len(label) else np.array([int(np.argmax(probability))]) for label, probability in zip(labels, probabilities)]
        return labels, optimal_count

    @staticmethod
    def _cluster_word_count(chunks: list[Chunk]) -> int:
        return sum(len(chunk.text.split()) for chunk in chunks)


class AgglomerativeRaptorTreeBuilder(RaptorTreeBuilder):
    """Laitenberger-style agglomerative RAPTOR with positional features and a root node."""

    def build(self, leaves: list[Chunk]) -> list[Chunk]:
        if len(leaves) <= 1:
            return []
        try:
            embeddings = _encode_texts(self.embedder, [chunk.text for chunk in leaves]) if self.embedder is not None else None
        except Exception:
            embeddings = None
        if embeddings is None:
            self.last_backend = "agglomerative_failed_contiguous"
            return super().build(leaves)

        first_cluster_count = max(1, int(np.ceil(len(leaves) / 3)))
        second_cluster_count = max(1, int(np.ceil(len(leaves) / 6)))
        first_labels = self._cluster_labels(embeddings, first_cluster_count)
        second_labels = self._cluster_labels(embeddings, second_cluster_count)

        level1: list[tuple[Chunk, set[int]]] = []
        for label in sorted(set(first_labels.tolist())):
            leaf_indices = {index for index, value in enumerate(first_labels) if value == label}
            cluster = [leaves[index] for index in sorted(leaf_indices)]
            if not cluster:
                continue
            level1.append((self._parent_chunk(leaves[0], 1, len(level1), cluster), leaf_indices))

        level2: list[Chunk] = []
        for label in sorted(set(second_labels.tolist())):
            leaf_indices = {index for index, value in enumerate(second_labels) if value == label}
            children = [parent for parent, child_indices in level1 if child_indices & leaf_indices]
            if not children:
                children = [leaves[index] for index in sorted(leaf_indices)]
            level2.append(self._parent_chunk(leaves[0], 2, len(level2), children))

        root = self._parent_chunk(leaves[0], 3, 0, level2 if level2 else [parent for parent, _indices in level1])
        self.last_backend = "agglomerative_dendrogram_n3_n6_root"
        return [parent for parent, _indices in level1] + level2 + [root]

    def _cluster_labels(self, embeddings: np.ndarray, cluster_count: int) -> np.ndarray:
        if cluster_count <= 1:
            return np.ones(len(embeddings), dtype=int)
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        features = self._position_augmented_embeddings(embeddings)
        distances = pdist(features, metric="cosine")
        tree = linkage(distances, method="average")
        return fcluster(tree, t=cluster_count, criterion="maxclust")

    def _position_augmented_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        count = len(embeddings)
        if count <= 1:
            return embeddings
        positions = np.arange(count, dtype=np.float32)
        start = (positions / max(count - 1, 1) - 0.5) * 0.2
        distance_to_end = ((count - 1 - positions) / max(count - 1, 1) - 0.5) * 0.2
        section = ((np.floor(positions / max(count, 1) * 3) / 2.0) - 0.5) * 0.2
        position_features = np.vstack([start, distance_to_end, section]).T * self.config.position_priority
        return np.hstack([embeddings, position_features])

    def _parent_chunk(self, first: Chunk, level: int, index: int, children: list[Chunk]) -> Chunk:
        return Chunk(
            chunk_id=f"{first.doc_id}::raptor::level{level}::{index}",
            doc_id=first.doc_id,
            title=first.title,
            section=f"raptor_level_{level}",
            text=self._summarize(children),
        )

    def _groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if self.embedder is None or len(chunks) <= self.config.group_size:
            self.last_backend = "contiguous"
            return self._contiguous_groups(chunks)
        try:
            embeddings = _encode_texts(self.embedder, [chunk.text for chunk in chunks])
            from sklearn.cluster import AgglomerativeClustering
        except Exception:
            self.last_backend = "agglomerative_failed_contiguous"
            return self._contiguous_groups(chunks)

        cluster_count = min(max(2, int(np.ceil(len(chunks) / self.config.group_size))), len(chunks))
        positions = np.linspace(0.0, 1.0, len(chunks), dtype=np.float32).reshape(-1, 1)
        features = np.hstack([embeddings, positions * self.config.position_priority * 0.02])
        try:
            model = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
            labels = model.fit_predict(features)
        except Exception:
            self.last_backend = "agglomerative_failed_contiguous"
            return self._contiguous_groups(chunks)

        groups: list[list[int]] = []
        for label in sorted(set(int(value) for value in labels)):
            indices = [index for index, value in enumerate(labels) if int(value) == label]
            groups.extend(self._split_indices(indices))
        self.last_backend = "agglomerative_position"
        return [[chunks[index] for index in group] for group in groups if group]


class RaptorGMMAbstractivePipeline:
    """RAPTOR closer to the original paper: GMM clustering, abstractive parents, collapsed retrieval."""

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
        self.tree_builder = GMMRaptorTreeBuilder(
            RaptorConfig(group_size=group_size, use_leiden=False),
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


class RaptorAgglomerativeAbstractivePipeline:
    """Position-aware agglomerative RAPTOR for deeper hierarchy experiments."""

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
        self.tree_builder = AgglomerativeRaptorTreeBuilder(
            RaptorConfig(group_size=group_size, max_levels=3, use_leiden=False),
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


class SemanticRaptorLeidenRerankerPipeline:
    """Semantic leaves, adaptive Leiden RAPTOR parents, collapsed retrieval, and reranking."""

    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_words: int = 60,
        max_words: int = 220,
        breakpoint_threshold: float = 0.30,
        group_size: int = 4,
        retrieve_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.summarizer = AbstractiveClusterSummarizer(self.generator)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(
                min_words=min_words,
                max_words=max_words,
                breakpoint_threshold=breakpoint_threshold,
            ),
            embedder=self.retriever.model,
        )
        self.tree_builder = RaptorTreeBuilder(
            RaptorConfig(
                group_size=group_size,
                use_leiden=True,
                adaptive_threshold=True,
                semantic_similarity_threshold=1.0 - breakpoint_threshold,
                random_state=224,
            ),
            embedder=self.retriever.model,
            summarizer=self.summarizer,
        )
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.parent_count = 0
        self.raptor_backend = "not_run"

    def index_document(self, record: dict[str, Any]) -> None:
        leaves = build_semantic_document_chunks(record, chunker=self.chunker)
        parents = self.tree_builder.build(leaves)
        self.parent_count = len(parents)
        self.raptor_backend = self.tree_builder.last_backend
        self.retriever.index([*leaves, *parents])

    def answer(self, question: str) -> dict[str, Any]:
        candidates = self.retriever.search(question, top_k=self.retrieve_k)
        reranked = self.reranker.rerank(question, candidates, top_k=self.top_k)
        contexts = [chunk for chunk, _score in reranked]
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score for _chunk, score in reranked],
            "raptor_parent_count": self.parent_count,
            "raptor_backend": self.raptor_backend,
            "reranker_model": self.reranker.model_name,
            "reranker_load_error": self.reranker.load_error,
        }


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
