from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = ROOT / "notebooks" / "independent_variants" / "qasper_base_dense_standalone.ipynb"
OUTPUT_DIR = ROOT / "notebooks" / "independent_variants"

BASE_SETUP_CELL = '''# Simple Kaggle/Colab setup. Run this cell first.
# Do not force reinstall Kaggle's scientific stack; only install packages if missing.
import importlib.metadata as importlib_metadata
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "datasets": "datasets",
    "pyarrow": "pyarrow",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "torch": "torch",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "pandas": "pandas",
    "tqdm": "tqdm",
}

missing = [package for module, package in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import sklearn
import torch
from sentence_transformers import SentenceTransformer

def version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"

print("Dependency check OK:")
print("python", sys.version.split()[0])
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("torch", torch.__version__)
print("transformers", version("transformers"))
print("sentence-transformers", version("sentence-transformers"))
'''

LEIDEN_SETUP_CELL = '''# Simple Kaggle/Colab setup. Run this cell first.
# Do not force reinstall Kaggle's scientific stack; only install packages if missing.
import importlib.metadata as importlib_metadata
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "datasets": "datasets",
    "pyarrow": "pyarrow",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "torch": "torch",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "igraph": "igraph",
    "leidenalg": "leidenalg",
}

missing = [package for module, package in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import sklearn
import torch
from sentence_transformers import SentenceTransformer

def version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"

print("Dependency check OK:")
print("python", sys.version.split()[0])
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("torch", torch.__version__)
print("transformers", version("transformers"))
print("sentence-transformers", version("sentence-transformers"))
print("igraph", version("igraph"))
print("leidenalg", version("leidenalg"))
'''


VARIANTS = {
    "semantic_chunking_dense": {
        "filename": "qasper_semantic_chunking_dense_standalone.ipynb",
        "title": "Qasper semantic_chunking_dense Standalone",
        "description": "Semantic sentence-boundary chunking plus dense retrieval. This isolates preprocessing/chunking from retrieval and generation.",
    },
    "semantic_chunking_reranker": {
        "filename": "qasper_semantic_chunking_reranker_standalone.ipynb",
        "title": "Qasper semantic_chunking_reranker Standalone",
        "description": "Semantic sentence-boundary chunking plus dense candidate retrieval followed by a cross-encoder reranker.",
    },
    "semantic_chunking_hybrid_reranker": {
        "filename": "qasper_semantic_chunking_hybrid_reranker_standalone.ipynb",
        "title": "Qasper semantic_chunking_hybrid_reranker Standalone",
        "description": "Semantic sentence-boundary chunking plus dense+BM25 RRF candidate retrieval followed by a cross-encoder reranker. Tune RETRIEVE_K and TOP_K in the config cell.",
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
    "raptor_gmm_abstractive": {
        "filename": "qasper_raptor_gmm_abstractive_standalone.ipynb",
        "title": "Qasper raptor_gmm_abstractive Standalone",
        "description": "Original RAPTOR-style GMM clustering with abstractive parent summaries and collapsed-tree retrieval.",
    },
    "raptor_leiden_abstractive": {
        "filename": "qasper_raptor_leiden_abstractive_standalone.ipynb",
        "title": "Qasper raptor_leiden_abstractive Standalone",
        "description": "More faithful RAPTOR: recursive abstractive parent summaries plus Leiden graph clustering when igraph/leidenalg are available.",
    },
    "raptor_agglomerative_abstractive": {
        "filename": "qasper_raptor_agglomerative_abstractive_standalone.ipynb",
        "title": "Qasper raptor_agglomerative_abstractive Standalone",
        "description": "Position-aware agglomerative RAPTOR with abstractive parent summaries for deeper hierarchy experiments.",
    },
    "semantic_raptor_leiden_reranker": {
        "filename": "qasper_semantic_raptor_leiden_reranker_standalone.ipynb",
        "title": "Qasper semantic_raptor_leiden_reranker Standalone",
        "description": "Semantic chunking plus adaptive Leiden RAPTOR collapsed-tree retrieval followed by cross-encoder reranking.",
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


class SemanticRerankerPipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, reranker_model: str, min_words: int, max_words: int, breakpoint_threshold: float, retrieve_k: int, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(min_words=min_words, max_words=max_words, breakpoint_threshold=breakpoint_threshold),
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
    def __init__(self, *, retriever_model: str, generator_model: str, reranker_model: str, min_words: int, max_words: int, breakpoint_threshold: float, retrieve_k: int, top_k: int) -> None:
        self.dense = DenseRetriever(retriever_model)
        self.sparse = BM25Retriever()
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(min_words=min_words, max_words=max_words, breakpoint_threshold=breakpoint_threshold),
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
        with torch.inference_mode():
            outputs = self.generator.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=1)
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
                threshold = self._similarity_threshold(embeddings)
                for neighbour in np.where(similarities >= threshold)[0].tolist():
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
        groups = []
        for community in partition:
            community_indices = sorted(int(index) for index in community)
            groups.extend([community_indices[i:i + self.config.group_size] for i in range(0, len(community_indices), self.config.group_size)])
        self.last_backend = "leiden"
        return [[chunks[index] for index in group] for group in groups if group]

    def _contiguous_groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[index:index + self.config.group_size] for index in range(0, len(chunks), self.config.group_size)]

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
        sentences = []
        for chunk in chunks:
            chunk_sentences = split_sentences(chunk.text)
            if chunk_sentences:
                sentences.append(chunk_sentences[0])
        return " ".join(" ".join(sentences).split()[: self.config.max_summary_words])


class GMMRaptorTreeBuilder(RaptorTreeBuilder):
    def _groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if self.embedder is None or len(chunks) <= self.config.group_size:
            self.last_backend = "contiguous"
            return self._contiguous_groups(chunks)
        try:
            embeddings = encode_texts(self.embedder, [chunk.text for chunk in chunks])
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
        grouped = []
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
                if sum(len(chunk.text.split()) for chunk in cluster) > self.config.max_cluster_words and len(cluster) > 1:
                    grouped.extend(self._raptor_clusters(cluster, embeddings[indices]))
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


class AgglomerativeRaptorTreeBuilder(RaptorTreeBuilder):
    def build(self, leaves: list[Chunk]) -> list[Chunk]:
        if len(leaves) <= 1:
            return []
        try:
            embeddings = encode_texts(self.embedder, [chunk.text for chunk in leaves]) if self.embedder is not None else None
        except Exception:
            embeddings = None
        if embeddings is None:
            self.last_backend = "agglomerative_failed_contiguous"
            return super().build(leaves)

        first_cluster_count = max(1, int(np.ceil(len(leaves) / 3)))
        second_cluster_count = max(1, int(np.ceil(len(leaves) / 6)))
        first_labels = self._cluster_labels(embeddings, first_cluster_count)
        second_labels = self._cluster_labels(embeddings, second_cluster_count)
        level1 = []
        for label in sorted(set(first_labels.tolist())):
            leaf_indices = {index for index, value in enumerate(first_labels) if value == label}
            cluster = [leaves[index] for index in sorted(leaf_indices)]
            if cluster:
                level1.append((self._parent_chunk(leaves[0], 1, len(level1), cluster), leaf_indices))
        level2 = []
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
            f"{first.doc_id}::raptor::level{level}::{index}",
            first.doc_id,
            first.title,
            f"raptor_level_{level}",
            self._summarize(children),
        )

    def _groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if self.embedder is None or len(chunks) <= self.config.group_size:
            self.last_backend = "contiguous"
            return self._contiguous_groups(chunks)
        try:
            embeddings = encode_texts(self.embedder, [chunk.text for chunk in chunks])
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
        groups = []
        for label in sorted(set(int(value) for value in labels)):
            indices = [index for index, value in enumerate(labels) if int(value) == label]
            groups.extend([indices[i:i + self.config.group_size] for i in range(0, len(indices), self.config.group_size)])
        self.last_backend = "agglomerative_position"
        return [[chunks[index] for index in group] for group in groups if group]


class RaptorGMMAbstractivePipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, group_size: int, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.summarizer = AbstractiveClusterSummarizer(self.generator)
        self.tree_builder = GMMRaptorTreeBuilder(RaptorConfig(group_size=group_size, use_leiden=False), embedder=self.retriever.model, summarizer=self.summarizer)
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
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score_by_id[chunk.chunk_id] for chunk in contexts], "raptor_parent_count": self.parent_count, "raptor_backend": self.raptor_backend}


class RaptorAgglomerativeAbstractivePipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, chunk_size: int, overlap: int, group_size: int, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.summarizer = AbstractiveClusterSummarizer(self.generator)
        self.tree_builder = AgglomerativeRaptorTreeBuilder(RaptorConfig(group_size=group_size, max_levels=3, use_leiden=False), embedder=self.retriever.model, summarizer=self.summarizer)
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
        return {"answer": self.generator.answer(question, contexts), "contexts": contexts, "scores": [score_by_id[chunk.chunk_id] for chunk in contexts], "raptor_parent_count": self.parent_count, "raptor_backend": self.raptor_backend}


class SemanticRaptorLeidenRerankerPipeline:
    def __init__(self, *, retriever_model: str, generator_model: str, reranker_model: str, min_words: int, max_words: int, breakpoint_threshold: float, group_size: int, retrieve_k: int, top_k: int) -> None:
        self.retriever = DenseRetriever(retriever_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.summarizer = AbstractiveClusterSummarizer(self.generator)
        self.chunker = SemanticChunker(
            SemanticChunkingConfig(min_words=min_words, max_words=max_words, breakpoint_threshold=breakpoint_threshold),
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
    if variant == "semantic_chunking_reranker":
        return SemanticRerankerPipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, reranker_model=RERANKER_MODEL, min_words=SEMANTIC_MIN_WORDS, max_words=CHUNK_SIZE, breakpoint_threshold=SEMANTIC_BREAKPOINT_THRESHOLD, retrieve_k=RETRIEVE_K, top_k=TOP_K)
    if variant == "semantic_chunking_hybrid_reranker":
        return SemanticHybridRerankerPipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, reranker_model=RERANKER_MODEL, min_words=SEMANTIC_MIN_WORDS, max_words=CHUNK_SIZE, breakpoint_threshold=SEMANTIC_BREAKPOINT_THRESHOLD, retrieve_k=RETRIEVE_K, top_k=TOP_K)
    if variant == "dense_reranker":
        return DenseRerankerPipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, reranker_model=RERANKER_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, retrieve_k=RETRIEVE_K, top_k=TOP_K)
    if variant == "raptor_extractive":
        return RaptorExtractivePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, group_size=RAPTOR_GROUP_SIZE, top_k=TOP_K)
    if variant == "raptor_gmm_abstractive":
        return RaptorGMMAbstractivePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, group_size=RAPTOR_GROUP_SIZE, top_k=TOP_K)
    if variant == "raptor_leiden_abstractive":
        return RaptorLeidenAbstractivePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, group_size=RAPTOR_GROUP_SIZE, top_k=TOP_K)
    if variant == "raptor_agglomerative_abstractive":
        return RaptorAgglomerativeAbstractivePipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP, group_size=RAPTOR_GROUP_SIZE, top_k=TOP_K)
    if variant == "semantic_raptor_leiden_reranker":
        return SemanticRaptorLeidenRerankerPipeline(retriever_model=RETRIEVER_MODEL, generator_model=GENERATOR_MODEL, reranker_model=RERANKER_MODEL, min_words=SEMANTIC_MIN_WORDS, max_words=CHUNK_SIZE, breakpoint_threshold=0.30, group_size=RAPTOR_GROUP_SIZE, retrieve_k=RETRIEVE_K, top_k=TOP_K)
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
    index_seconds_total = 0.0
    answer_seconds_total = 0.0
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
            "index_seconds_total": index_seconds_total,
            "index_seconds_per_doc": index_seconds_total / docs_seen if docs_seen else 0.0,
            "answer_seconds_total": answer_seconds_total,
            "answer_seconds_per_example": answer_seconds_total / rows if rows else 0.0,
            "config": CONFIG,
            "metrics": metrics,
            "predictions_path": str(predictions_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    with predictions_path.open("w", encoding="utf-8") as file:
        for record in tqdm(selected_records(dataset, min_doc_words=MIN_DOC_WORDS), desc=f"Running {VARIANT}"):
            docs_seen += 1
            index_start = time.perf_counter()
            pipeline.index_document(record)
            index_seconds_total += time.perf_counter() - index_start
            for example in extract_qa_examples(record):
                answer_start = time.perf_counter()
                answer_result = pipeline.answer(example.question)
                answer_seconds = time.perf_counter() - answer_start
                answer_seconds_total += answer_seconds
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
                    "answer_seconds": answer_seconds,
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
    notebook["cells"][0]["source"] = source_lines(
        f"# {meta['title']}\n\n"
        f"{meta['description']}\n\n"
        "This standalone notebook contains all code needed to run on Kaggle/Colab. "
        "It does not clone the repo and does not import from `src/`. "
        "By default it only runs papers with `MIN_DOC_WORDS >= 3000` to focus on long-context cases.\n"
    )
    notebook["cells"][2]["source"] = source_lines(CONFIG_TEMPLATE.format(variant=variant))
    notebook["cells"][7]["source"] = source_lines(PIPELINES_CODE)
    notebook["cells"][8]["source"] = source_lines(RUN_CODE)
    if variant in {"raptor_leiden_abstractive", "semantic_raptor_leiden_reranker"}:
        notebook["cells"][1]["source"] = source_lines(LEIDEN_SETUP_CELL)
    else:
        notebook["cells"][1]["source"] = source_lines(BASE_SETUP_CELL)
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
