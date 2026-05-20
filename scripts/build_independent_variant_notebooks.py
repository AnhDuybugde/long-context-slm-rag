from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = ROOT / "notebooks" / "independent_variants" / "qasper_base_dense_standalone.ipynb"
OUTPUT_DIR = ROOT / "notebooks" / "independent_variants"


VARIANTS = {
    "semantic_chunking_dense": {
        "filename": "qasper_semantic_chunking_dense_standalone.ipynb",
        "title": "Qasper semantic_chunking_dense Standalone",
        "description": "Semantic sentence-boundary chunking plus dense retrieval. This isolates preprocessing/chunking from retrieval and generation.",
    },
    "dense_reranker": {
        "filename": "qasper_dense_reranker_standalone.ipynb",
        "title": "Qasper dense_reranker Standalone",
        "description": "Dense retrieval followed by a cross-encoder reranker. If the reranker cannot load, the notebook falls back to a lexical rerank score and records the load error.",
    },
    "raptor_extractive": {
        "filename": "qasper_raptor_extractive_standalone.ipynb",
        "title": "Qasper raptor_extractive Standalone",
        "description": "RAPTOR-style collapsed tree with extractive parent summaries and graph-similarity grouping. This is an offline proxy, not full LLM-abstractive RAPTOR.",
    },
    "raptor_leiden_abstractive": {
        "filename": "qasper_raptor_leiden_abstractive_standalone.ipynb",
        "title": "Qasper raptor_leiden_abstractive Standalone",
        "description": "More faithful RAPTOR: recursive abstractive parent summaries plus Leiden graph clustering when igraph/leidenalg are available.",
    },
}


CONFIG_TEMPLATE = '''VARIANT = "{variant}"

SPLIT = "validation"
MIN_DOC_WORDS = 3000
LIMIT = None  # Set to 10 for a smoke test.
TOP_K = 5
RETRIEVE_K = 20
CHUNK_SIZE = 180
OVERLAP = 40
SEMANTIC_MIN_WORDS = 60
SEMANTIC_BREAKPOINT_THRESHOLD = 0.35
RAPTOR_GROUP_SIZE = 4
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-base"
OUTPUT_DIR = "outputs/independent"

CONFIG = {{
    "variant": VARIANT,
    "split": SPLIT,
    "min_doc_words": MIN_DOC_WORDS,
    "limit": LIMIT,
    "top_k": TOP_K,
    "retrieve_k": RETRIEVE_K,
    "chunk_size": CHUNK_SIZE,
    "overlap": OVERLAP,
    "semantic_min_words": SEMANTIC_MIN_WORDS,
    "semantic_breakpoint_threshold": SEMANTIC_BREAKPOINT_THRESHOLD,
    "raptor_group_size": RAPTOR_GROUP_SIZE,
    "reranker_model": RERANKER_MODEL,
    "retriever_model": RETRIEVER_MODEL,
    "generator_model": GENERATOR_MODEL,
}}
CONFIG
'''


PIPELINES_CODE = r'''def token_overlap_recall(candidate: str, reference: str) -> float:
    candidate_tokens = set(normalize_text(candidate))
    reference_tokens = set(normalize_text(reference))
    if not reference_tokens:
        return 0.0
    return len(candidate_tokens & reference_tokens) / len(reference_tokens)


class BaseDensePipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        self.retriever.index(build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap))

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [chunk for chunk, _score in retrieved]
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score for _chunk, score in retrieved]}


class BM25OnlyPipeline:
    def __init__(self, *, generator_model: str, chunk_size: int, overlap: int, top_k: int) -> None:
        self.retriever = BM25Retriever()
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        self.retriever.index(build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap))

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [chunk for chunk, _score in retrieved]
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score for _chunk, score in retrieved]}


class DenseReorderPipeline(BaseDensePipeline):
    def __init__(self, *, reorder_mode: str, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, top_k: int) -> None:
        super().__init__(retriever_model=retriever_model, generator_model=generator_model, chunk_size=chunk_size, overlap=overlap, top_k=top_k)
        self.reorder_mode = reorder_mode

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        score_by_id = {chunk.chunk_id: score for chunk, score in retrieved}
        contexts = [chunk for chunk, _score in retrieved]
        if self.reorder_mode == "u_shape":
            contexts = u_shaped_reorder(contexts)
        elif self.reorder_mode == "recency_heavy":
            contexts = recency_heavy_reorder(contexts)
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score_by_id[chunk.chunk_id] for chunk in contexts]}


class HybridRRFPipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, retrieve_k: int, top_k: int) -> None:
        self.dense = DenseRetriever(retriever_model)
        self.sparse = BM25Retriever()
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retrieve_k = retrieve_k
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        chunks = build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap)
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def answer(self, question: str) -> dict[str, Any]:
        dense_results = self.dense.search(question, top_k=self.retrieve_k)
        sparse_results = self.sparse.search(question, top_k=self.retrieve_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=self.top_k)
        contexts = [chunk for chunk, _score in fused]
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score for _chunk, score in fused]}


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def encode_texts(embedder, texts: list[str]) -> np.ndarray:
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
    def __init__(self, config: SemanticChunkingConfig | None = None, *, embedder=None) -> None:
        self.config = config or SemanticChunkingConfig()
        self.embedder = embedder

    def chunk_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if self.embedder is None or len(sentences) == 1:
            return chunk_words(text, chunk_size=self.config.max_words, overlap=0)
        try:
            embeddings = encode_texts(self.embedder, sentences)
        except Exception:
            return chunk_words(text, chunk_size=self.config.max_words, overlap=0)

        chunks: list[str] = []
        current = [sentences[0]]
        current_words = len(sentences[0].split())
        for index in range(1, len(sentences)):
            sentence = sentences[index]
            sentence_words = len(sentence.split())
            distance = 1.0 - float(np.dot(embeddings[index - 1], embeddings[index]))
            too_large = current_words + sentence_words > self.config.max_words
            semantic_break = current_words >= self.config.min_words and distance >= self.config.breakpoint_threshold
            if too_large or semantic_break:
                chunks.append(" ".join(current).strip())
                overlap = current[-self.config.overlap_sentences:] if self.config.overlap_sentences else []
                current = [*overlap, sentence]
                current_words = sum(len(item.split()) for item in current)
            else:
                current.append(sentence)
                current_words += sentence_words
        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]


def build_semantic_document_chunks(record: dict[str, Any], *, chunker: SemanticChunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for text in chunker.chunk_text(str(record.get("abstract", "")).strip()):
        chunks.append(Chunk(f"{record['id']}::semantic::abstract::{chunk_index}", record["id"], record.get("title", ""), "abstract", text))
        chunk_index += 1
    full_text = record.get("full_text", {})
    for section, paragraphs in zip(full_text.get("section_name", []), full_text.get("paragraphs", [])):
        section_text = " ".join(str(paragraph) for paragraph in paragraphs if str(paragraph).strip())
        for text in chunker.chunk_text(section_text):
            chunks.append(Chunk(f"{record['id']}::semantic::{chunk_index}", record["id"], record.get("title", ""), str(section), text))
            chunk_index += 1
    return chunks


class SemanticDensePipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, min_words: int, max_words: int, breakpoint_threshold: float, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(min_words=min_words, max_words=max_words, breakpoint_threshold=breakpoint_threshold),
            embedder=self.retriever.model,
        )
        self.top_k = top_k

    def index_document(self, record: dict[str, Any]) -> None:
        self.retriever.index(build_semantic_document_chunks(record, chunker=self.chunker))

    def answer(self, question: str) -> dict[str, Any]:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [chunk for chunk, _score in retrieved]
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score for _chunk, score in retrieved]}


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.load_error = None
        if model_name is None:
            self.load_error = "disabled"
            return
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception as error:
            self.load_error = str(error)

    def rerank(self, question: str, candidates: list[tuple[Chunk, float]], *, top_k: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        if self.model is not None:
            pairs = [(question, chunk.text) for chunk, _score in candidates]
            scores = [float(score) for score in self.model.predict(pairs)]
        else:
            scores = [float(original_score) + token_overlap_recall(chunk.text, question) for chunk, original_score in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [(chunk, score) for ((chunk, _original_score), score) in ranked]


class DenseRerankerPipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, reranker_model: str, chunk_size: int, overlap: int, retrieve_k: int, top_k: int) -> None:
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
    def __init__(self, generator: SmallSeq2SeqGenerator, *, max_input_tokens: int = 768, max_new_tokens: int = 96) -> None:
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
        inputs = self.generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.generator.device)
        outputs = self.generator.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=2)
        return self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class RaptorTreeBuilder:
    def __init__(self, config: RaptorConfig | None = None, *, embedder=None, summarizer=None) -> None:
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
                first = group[0]
                parent = Chunk(
                    f"{first.doc_id}::raptor::level{level}::{group_index}",
                    first.doc_id,
                    first.title,
                    f"raptor_level_{level}",
                    self._summarize(group),
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
            embeddings = encode_texts(self.embedder, [chunk.text for chunk in chunks])
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
                for neighbour in np.where(similarities >= self.config.similarity_threshold)[0].tolist():
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
            groups.extend([sorted(component)[i:i + self.config.group_size] for i in range(0, len(component), self.config.group_size)])
        return [[chunks[index] for index in group] for group in groups]

    def _leiden_groups(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[list[Chunk]]:
        import igraph as ig
        import leidenalg

        edges = []
        weights = []
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
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights=graph.es["weight"])
        groups = []
        for community in partition:
            community_indices = sorted(int(index) for index in community)
            groups.extend([community_indices[i:i + self.config.group_size] for i in range(0, len(community_indices), self.config.group_size)])
        self.last_backend = "leiden"
        return [[chunks[index] for index in group] for group in groups if group]

    def _contiguous_groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[index:index + self.config.group_size] for index in range(0, len(chunks), self.config.group_size)]

    def _summarize(self, chunks: list[Chunk]) -> str:
        if self.summarizer is not None:
            summary = self.summarizer.summarize(chunks).strip()
            if summary:
                return summary
        sentences = []
        for chunk in chunks:
            chunk_sentences = split_sentences(chunk.text)
            if chunk_sentences:
                sentences.append(chunk_sentences[0])
        return " ".join(" ".join(sentences).split()[: self.config.max_summary_words])


class RaptorExtractivePipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, group_size: int, top_k: int) -> None:
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
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score_by_id[chunk.chunk_id] for chunk in contexts]}


class RaptorLeidenAbstractivePipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, group_size: int, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.summarizer = AbstractiveClusterSummarizer(self.generator)
        self.tree_builder = RaptorTreeBuilder(RaptorConfig(group_size=group_size, use_leiden=True), embedder=self.retriever.model, summarizer=self.summarizer)
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


def build_pipeline(variant: str):
    if variant == "base_dense":
        return BaseDensePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, top_k=TOP_K)
    if variant == "bm25_only":
        return BM25OnlyPipeline(generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, top_k=TOP_K)
    if variant == "dense_u_shape":
        return DenseReorderPipeline(reorder_mode="u_shape", retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, top_k=TOP_K)
    if variant == "dense_recency_heavy":
        return DenseReorderPipeline(reorder_mode="recency_heavy", retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, top_k=TOP_K)
    if variant == "hybrid_rrf":
        return HybridRRFPipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, retrieve_k=RETRIEVE_K, top_k=TOP_K)
    if variant == "semantic_chunking_dense":
        return SemanticDensePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, min_words=SEMANTIC_MIN_WORDS, max_words=CHUNK_SIZE, breakpoint_threshold=SEMANTIC_BREAKPOINT_THRESHOLD, top_k=TOP_K)
    if variant == "dense_reranker":
        return DenseRerankerPipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, reranker_model=RERANKER_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, retrieve_k=RETRIEVE_K, top_k=TOP_K)
    if variant == "raptor_extractive":
        return RaptorExtractivePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, group_size=RAPTOR_GROUP_SIZE, top_k=TOP_K)
    if variant == "raptor_leiden_abstractive":
        return RaptorLeidenAbstractivePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, group_size=RAPTOR_GROUP_SIZE, top_k=TOP_K)
    raise ValueError(f"Unknown variant: {variant}")
'''


RUN_CODE = r'''def selected_records(dataset, *, min_doc_words: int):
    for record in dataset:
        if min_doc_words <= 0 or document_word_count(record) >= min_doc_words:
            yield record


def serialize_contexts(contexts: list[Chunk], scores: list[float]) -> list[dict[str, Any]]:
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


def survey_dataset(dataset) -> dict[str, Any]:
    lengths = [document_word_count(record) for record in dataset]
    lengths_sorted = sorted(lengths)
    thresholds = [1000, 3000, 5000, 8000, 12000]
    return {
        "documents": len(lengths),
        "word_count_min": min(lengths_sorted),
        "word_count_median": int(median(lengths_sorted)),
        "word_count_mean": mean(lengths_sorted),
        "word_count_p90": lengths_sorted[round((len(lengths_sorted) - 1) * 0.90)],
        "word_count_max": max(lengths_sorted),
        "thresholds": {threshold: sum(1 for value in lengths if value >= threshold) for threshold in thresholds},
    }


def run_experiment(dataset) -> dict[str, Any]:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{VARIANT}_{SPLIT}_min{MIN_DOC_WORDS}_predictions.jsonl"
    summary_path = output_dir / f"{VARIANT}_{SPLIT}_min{MIN_DOC_WORDS}_summary.json"

    pipeline = build_pipeline(VARIANT)
    totals = Counter()
    rows = 0
    docs_seen = 0
    start = time.perf_counter()

    def write_summary() -> dict[str, Any]:
        runtime = time.perf_counter() - start
        metrics = {"examples": rows, **{f"avg_{key}": value / rows for key, value in totals.items()}} if rows else {"examples": 0}
        summary = {
            "variant": VARIANT,
            "split": SPLIT,
            "min_doc_words": MIN_DOC_WORDS,
            "docs_seen": docs_seen,
            "runtime_seconds": runtime,
            "seconds_per_example": runtime / rows if rows else 0.0,
            "config": CONFIG,
            "metrics": metrics,
            "predictions_path": str(predictions_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    with predictions_path.open("w", encoding="utf-8") as file:
        for record in tqdm(selected_records(dataset, min_doc_words=MIN_DOC_WORDS), desc=f"Running {VARIANT}"):
            docs_seen += 1
            pipeline.index_document(record)
            for example in extract_qa_examples(record):
                answer_result = pipeline.answer(example.question)
                contexts = answer_result["contexts"]
                scores = answer_result["scores"]
                prediction = answer_result["answer"]
                extra = {key: value for key, value in answer_result.items() if key not in {"answer", "contexts", "scores"}}
                row_metrics = {
                    "token_f1": best_f1(prediction, example.gold_answers),
                    f"answer_string_recall_at_{TOP_K}": answer_string_recall(contexts, example.gold_answers),
                    "context_precision": context_precision(contexts, example.gold_answers, example.evidence),
                    "context_recall": context_recall(contexts, example.gold_answers, example.evidence),
                    "faithfulness": faithfulness(prediction, contexts),
                    "answer_relevancy": answer_relevancy(prediction, example.question, example.gold_answers),
                }
                row = {
                    "doc_id": example.doc_id,
                    "question_id": example.question_id,
                    "title": example.title,
                    "question": example.question,
                    "prediction": prediction,
                    "gold_answers": example.gold_answers,
                    "evidence": example.evidence,
                    "metrics": row_metrics,
                    "contexts": serialize_contexts(contexts, scores),
                    **extra,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                totals.update(row_metrics)
                rows += 1
                if LIMIT is not None and rows >= LIMIT:
                    return write_summary()

    return write_summary()
'''


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build_notebook(base: dict, *, variant: str, meta: dict[str, str]) -> dict:
    notebook = copy.deepcopy(base)
    notebook["cells"][0]["source"] = source_lines(f"# {meta['title']}\n\n{meta['description']}\n\nNotebook nay tu chua toan bo code de chay tren Kaggle/Colab. Khong clone repo, khong import tu `src/`. Mac dinh chi chay paper co `MIN_DOC_WORDS >= 3000` de tap trung vao long-context.\n")
    notebook["cells"][2]["source"] = source_lines(CONFIG_TEMPLATE.format(variant=variant))
    notebook["cells"][7]["source"] = source_lines(PIPELINES_CODE)
    notebook["cells"][8]["source"] = source_lines(RUN_CODE)
    if variant == "raptor_leiden_abstractive":
        notebook["cells"][1]["source"] = source_lines(
            '# Kaggle/Colab setup. Run once per fresh session.\n'
            '!pip -q install -U "datasets>=2.19.0" "pyarrow>=15.0.0" "sentence-transformers>=2.7.0" "transformers>=4.41.0" "torch>=2.2.0" "numpy>=1.26.0" "tqdm>=4.66.0" "igraph>=0.11.0" "leidenalg>=0.10.0"\n'
        )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


def main() -> None:
    base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    for variant, meta in VARIANTS.items():
        notebook = build_notebook(base, variant=variant, meta=meta)
        output_path = OUTPUT_DIR / meta["filename"]
        output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)


if __name__ == "__main__":
    main()
