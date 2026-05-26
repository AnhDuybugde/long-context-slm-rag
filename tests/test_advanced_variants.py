import unittest

import numpy as np

from src.qasper_base_rag.advanced_variants import (
    AgglomerativeRaptorTreeBuilder,
    ContextualSemanticRerankerPipeline,
    CrossEncoderReranker,
    E5QwenFilterGeneratorPipeline,
    EvidenceSentenceSelector,
    GraphRagRaptorConfig,
    GraphRagRaptorNode,
    GraphRagRaptorTreeBuilder,
    GraphRagRaptorWideLateChunkingPipeline,
    GMMRaptorTreeBuilder,
    OracleGoldContextPromptAblationPipeline,
    OracleGoldContextPipeline,
    OraclePromptSeq2SeqGenerator,
    QwenEvidenceFilterCompressor,
    RaptorConfig,
    RaptorTreeBuilder,
    LateChunkingDenseRetriever,
    SemanticChunker,
    SemanticChunkSpan,
    SemanticChunkingConfig,
    SemanticRaptorLeidenRerankerPipeline,
    SemanticHybridRerankerPipeline,
    SemanticRerankerPipeline,
    SemanticRerankerQwenDirectPipeline,
    SentenceSelectWideLateChunkingPipeline,
    SelfRouteSemanticRerankerPipeline,
    SufficientContextGate,
    WideLateChunkingSemanticRerankerPipeline,
    build_semantic_document_chunk_spans,
    contextualize_chunk,
    oracle_question_overlap_score,
    oracle_tail_reminder_sentences,
    oracle_u_tail_reorder,
    short_document_summary,
)
from src.qasper_base_rag.chunking import Chunk
from src.qasper_base_rag.data import QAExample
from src.qasper_base_rag.experiment_pipelines import (
    GRAPH_RAPTOR_SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS,
    GRAPH_RAPTOR_WIDE_LATE_CHUNKING_CONFIGS,
    SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS,
    WIDE_LATE_CHUNKING_CONFIGS,
)


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def encode(self, texts, **_kwargs):
        return self.vectors[: len(texts)]


class AdvancedVariantsTest(unittest.TestCase):
    def test_semantic_chunker_breaks_on_large_cosine_distance(self):
        chunker = SemanticChunker(
            SemanticChunkingConfig(min_words=1, max_words=30, breakpoint_threshold=0.4, overlap_sentences=0),
            embedder=FakeEmbedder([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]),
        )

        chunks = chunker.chunk_text("Alpha method works. Alpha method continues. Beta result differs.")

        self.assertEqual(len(chunks), 2)
        self.assertIn("Alpha method continues.", chunks[0])
        self.assertIn("Beta result differs.", chunks[1])

    def test_wide_semantic_chunking_creates_larger_fewer_chunks(self):
        text = " ".join(f"Sentence {index} contains method evidence." for index in range(30))
        narrow = SemanticChunker(SemanticChunkingConfig(min_words=1, max_words=30, overlap_sentences=0))
        wide = SemanticChunker(SemanticChunkingConfig(min_words=1, max_words=90, overlap_sentences=0))

        narrow_chunks = narrow.chunk_text(text)
        wide_chunks = wide.chunk_text(text)

        self.assertLess(len(wide_chunks), len(narrow_chunks))
        self.assertGreater(max(len(chunk.split()) for chunk in wide_chunks), max(len(chunk.split()) for chunk in narrow_chunks))

    def test_semantic_chunk_spans_keep_source_metadata(self):
        record = {
            "id": "d1",
            "title": "Paper",
            "abstract": "Abstract evidence sentence.",
            "full_text": {
                "section_name": ["Method"],
                "paragraphs": [["The method uses wider chunks.", "Late chunking pools after encoding."]],
            },
        }
        chunker = SemanticChunker(SemanticChunkingConfig(min_words=1, max_words=20, overlap_sentences=0))

        spans = build_semantic_document_chunk_spans(record, chunker=chunker)

        self.assertTrue(spans)
        self.assertEqual(spans[0].chunk.section, "abstract")
        self.assertIn("semantic_source", spans[0].source_id)
        self.assertEqual(spans[0].source_text[spans[0].start_char : spans[0].end_char], spans[0].chunk.text)

    def test_late_chunking_pools_chunk_vectors_from_source_windows(self):
        spans = [
            SemanticChunkSpan(Chunk("c1", "d1", "Paper", "s", "alpha beta"), "s1", "alpha beta gamma", 0, 10),
            SemanticChunkSpan(Chunk("c2", "d1", "Paper", "s", "gamma"), "s1", "alpha beta gamma", 11, 16),
        ]
        hidden_states = np.asarray([[[1.0, 0.0], [3.0, 0.0], [0.0, 4.0], [9.0, 9.0]]], dtype=np.float32)
        offsets = np.asarray([[[0, 5], [6, 10], [11, 16], [0, 0]]])
        attention = np.asarray([[1, 1, 1, 0]])

        pooled = LateChunkingDenseRetriever._pool_spans_from_windows(hidden_states, offsets, attention, spans)

        np.testing.assert_allclose(pooled["c1"], np.asarray([2.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(pooled["c2"], np.asarray([0.0, 4.0], dtype=np.float32))

    def test_late_chunking_index_falls_back_when_span_missing(self):
        spans = [
            SemanticChunkSpan(Chunk("c1", "d1", "Paper", "s", "late vector"), "s1", "late vector fallback vector", 0, 11),
            SemanticChunkSpan(Chunk("c2", "d1", "Paper", "s", "fallback vector"), "s1", "late vector fallback vector", 12, 27),
        ]

        class FakeTextEmbedder:
            def encode(self, texts, **_kwargs):
                return np.asarray([[0.0, 5.0] for _text in texts], dtype=np.float32)

        retriever = LateChunkingDenseRetriever.__new__(LateChunkingDenseRetriever)
        retriever.model = FakeTextEmbedder()
        retriever.query_prefix = ""
        retriever.passage_prefix = "passage: "
        retriever.chunks = []
        retriever.embeddings = None
        retriever.late_chunking_window_count = 1
        retriever._late_encode_spans = lambda _spans: {"c1": np.asarray([4.0, 0.0], dtype=np.float32)}

        retriever.index_spans(spans)

        self.assertEqual(retriever.late_chunking_fallback_count, 1)
        self.assertEqual(retriever.late_chunking_backend, "late_chunking_with_fallback")
        np.testing.assert_allclose(retriever.embeddings[0], np.asarray([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(retriever.embeddings[1], np.asarray([0.0, 1.0], dtype=np.float32))

    def test_wide_late_variant_configs_set_expected_models_and_prefixes(self):
        minilm = WIDE_LATE_CHUNKING_CONFIGS["sem_rerank_minilm_wide_latechunk"]
        e5 = WIDE_LATE_CHUNKING_CONFIGS["sem_rerank_e5_wide_latechunk"]

        self.assertEqual(minilm["retriever_model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(minilm["query_prefix"], "")
        self.assertEqual(e5["retriever_model"], "intfloat/e5-base-v2")
        self.assertEqual(e5["query_prefix"], "query: ")
        self.assertEqual(e5["passage_prefix"], "passage: ")

    def test_graph_raptor_wide_late_configs_set_expected_models_and_prefixes(self):
        minilm = GRAPH_RAPTOR_WIDE_LATE_CHUNKING_CONFIGS["sem_rerank_minilm_wide_latechunk_graphrag_raptor"]
        e5 = GRAPH_RAPTOR_WIDE_LATE_CHUNKING_CONFIGS["sem_rerank_e5_wide_latechunk_graphrag_raptor"]

        self.assertEqual(minilm["retriever_model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(minilm["query_prefix"], "")
        self.assertEqual(e5["retriever_model"], "intfloat/e5-base-v2")
        self.assertEqual(e5["query_prefix"], "query: ")
        self.assertEqual(e5["passage_prefix"], "passage: ")

    def test_sentence_select_variant_configs_set_expected_models_and_prefixes(self):
        minilm = SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS["sem_rerank_minilm_wide_latechunk_sentence_select"]
        e5 = SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS["sem_rerank_e5_wide_latechunk_sentence_select"]
        graph_e5 = GRAPH_RAPTOR_SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS[
            "sem_rerank_e5_wide_latechunk_graphrag_raptor_sentence_select"
        ]

        self.assertEqual(minilm["retriever_model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(minilm["query_prefix"], "")
        self.assertEqual(e5["retriever_model"], "intfloat/e5-base-v2")
        self.assertEqual(e5["query_prefix"], "query: ")
        self.assertEqual(graph_e5["passage_prefix"], "passage: ")

    def test_evidence_sentence_selector_compresses_to_question_focused_sentence_window(self):
        contexts = [
            Chunk("c1", "d1", "Paper", "intro", "General background. The dataset used is Qasper. Extra details follow."),
            Chunk("c2", "d1", "Paper", "method", "Unrelated model architecture sentence."),
        ]
        selector = EvidenceSentenceSelector(max_sentences=1, window_sentences=1)

        selection = selector.select("Which dataset is used?", contexts, [0.9, 0.1])

        self.assertTrue(selection.sufficient)
        self.assertLess(selection.evidence_word_count, selection.source_word_count)
        self.assertEqual(selection.source_context_count, 2)
        self.assertTrue(any("Qasper" in chunk.text for chunk in selection.contexts))
        self.assertTrue(all(chunk.section.endswith("::evidence_sentence") for chunk in selection.contexts))

    def test_high_recall_sentence_selector_expands_complex_question_budget_with_source_diversity(self):
        contexts = [
            Chunk(
                "c1",
                "d1",
                "Paper",
                "data",
                "The datasets include Europarl. The datasets include MultiUN. The datasets include IWSLT. Extra background.",
            ),
            Chunk(
                "c2",
                "d1",
                "Paper",
                "metrics",
                "The metrics are BLEU. The metrics are accuracy. The metrics are F1.",
            ),
        ]
        selector = EvidenceSentenceSelector(
            max_sentences=1,
            window_sentences=0,
            high_recall=True,
            high_recall_max_sentences=2,
            high_recall_complex_max_sentences=4,
            max_sentences_per_context=2,
        )

        selection = selector.select("Which datasets and metrics are used?", contexts, [0.8, 0.7])

        self.assertGreaterEqual(selection.selected_sentence_count, 3)
        self.assertLessEqual(
            sum(1 for chunk in selection.contexts if chunk.chunk_id.startswith("c1::")),
            2,
        )
        self.assertTrue(any("BLEU" in chunk.text or "accuracy" in chunk.text for chunk in selection.contexts))

    def test_sentence_select_pipeline_abstains_when_sentence_evidence_is_weak(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "Neural machine translation background.")

        class FakeRetriever:
            model_name = "fake-retriever"
            query_prefix = ""
            passage_prefix = ""
            late_chunking_backend = "late_chunking"
            late_chunking_fallback_count = 0
            late_chunking_window_count = 1
            load_error = None

            def search(self, question, *, top_k):
                return [(c1, 0.1)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                return candidates

        class FakeGenerator:
            called = False

            def answer(self, question, contexts, *, tail_reminder_sentences=None):
                self.called = True
                return "should not run"

        pipeline = SentenceSelectWideLateChunkingPipeline.__new__(SentenceSelectWideLateChunkingPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 30
        pipeline.top_k = 1
        pipeline.query_prefix = ""
        pipeline.passage_prefix = ""
        pipeline.prompt_mode = "extractive"
        pipeline.context_order = "u_tail"
        pipeline.tail_reminder = True
        pipeline.min_words = 120
        pipeline.max_words = 420
        pipeline.breakpoint_threshold = 0.45
        pipeline.overlap_sentences = 2
        pipeline.late_max_tokens = 512
        pipeline.late_stride = 128
        pipeline.sentence_selector = EvidenceSentenceSelector(
            max_sentences=1,
            window_sentences=0,
            min_query_coverage=0.75,
            min_best_sentence_score=0.75,
        )
        pipeline.sentence_abstain_on_low_support = True

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Unanswerable")
        self.assertFalse(result["sentence_sufficient"])
        self.assertFalse(pipeline.generator.called)

    def test_graph_raptor_tree_builder_creates_parent_nodes_with_metadata(self):
        leaves = [
            Chunk("c1", "d1", "Paper", "method", "alpha method embeds related context"),
            Chunk("c2", "d1", "Paper", "method", "alpha method keeps semantic neighbours"),
            Chunk("c3", "d1", "Paper", "results", "beta result reports benchmark gains"),
            Chunk("c4", "d1", "Paper", "results", "beta result compares baselines"),
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
            dtype=np.float32,
        )
        builder = GraphRagRaptorTreeBuilder(
            GraphRagRaptorConfig(
                cluster_backend="agglomerative",
                max_levels=1,
                similarity_threshold=0.80,
                max_cluster_size=4,
            )
        )

        parents = builder.build(leaves, embeddings)

        self.assertEqual(len(parents), 2)
        self.assertTrue(all(parent.layer == 1 for parent in parents))
        self.assertTrue(all(parent.child_ids for parent in parents))
        self.assertTrue(all(parent.leaf_ids for parent in parents))
        self.assertTrue(all(parent.chunk.section == "graphrag_raptor_level_1" for parent in parents))

    def test_graph_raptor_routing_expands_only_selected_parent_branch(self):
        c1 = Chunk("c1", "d1", "Paper", "method", "alpha method evidence")
        c2 = Chunk("c2", "d1", "Paper", "method", "alpha context evidence")
        c3 = Chunk("c3", "d1", "Paper", "results", "beta unrelated evidence")
        c4 = Chunk("c4", "d1", "Paper", "results", "beta comparison evidence")
        p1 = Chunk("p1", "d1", "Paper", "graphrag_raptor_level_1", "alpha community summary")
        p2 = Chunk("p2", "d1", "Paper", "graphrag_raptor_level_1", "beta community summary")

        class FakeModel:
            def encode(self, texts, **_kwargs):
                return np.asarray([[1.0, 0.0] for _text in texts], dtype=np.float32)

        class FakeRetriever:
            model = FakeModel()
            embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)

            def search(self, question, *, top_k):
                return [(c3, 0.9)]

        pipeline = GraphRagRaptorWideLateChunkingPipeline.__new__(GraphRagRaptorWideLateChunkingPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.query_prefix = ""
        pipeline.ordered_chunks = [c1, c2, c3, c4]
        pipeline.graph_leaf_by_id = {chunk.chunk_id: chunk for chunk in pipeline.ordered_chunks}
        pipeline.graph_leaf_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(pipeline.ordered_chunks)}
        pipeline.graph_parent_nodes = [
            GraphRagRaptorNode(p1, np.asarray([1.0, 0.0], dtype=np.float32), 1, ("c1", "c2"), ("c1", "c2")),
            GraphRagRaptorNode(p2, np.asarray([0.0, 1.0], dtype=np.float32), 1, ("c3", "c4"), ("c3", "c4")),
        ]
        pipeline.graph_parent_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pipeline.graph_config = GraphRagRaptorConfig(
            tree_mode="local_tree",
            branch_k=1,
            parent_top_k=2,
            child_candidate_k=4,
            include_parent_context=False,
        )
        pipeline.graph_builder = GraphRagRaptorTreeBuilder(pipeline.graph_config)
        pipeline.graph_builder.last_backend = "test"

        candidates, metadata = pipeline._graph_candidates("Which alpha method is used?")

        self.assertEqual({chunk.chunk_id for chunk, _score in candidates}, {"c1", "c2"})
        self.assertEqual(metadata["graph_selected_parent_ids"], ["p1"])
        self.assertEqual(metadata["graph_route"], "tree")

    def test_graph_raptor_routing_falls_back_to_flat_when_no_parent_tree(self):
        c1 = Chunk("c1", "d1", "Paper", "method", "flat fallback evidence")

        class FakeModel:
            def encode(self, texts, **_kwargs):
                return np.asarray([[1.0, 0.0] for _text in texts], dtype=np.float32)

        class FakeRetriever:
            model = FakeModel()
            embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return [(c1, 0.7)]

        pipeline = GraphRagRaptorWideLateChunkingPipeline.__new__(GraphRagRaptorWideLateChunkingPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.query_prefix = ""
        pipeline.ordered_chunks = [c1]
        pipeline.graph_parent_nodes = []
        pipeline.graph_parent_embeddings = None
        pipeline.graph_config = GraphRagRaptorConfig(child_candidate_k=9)
        pipeline.graph_builder = GraphRagRaptorTreeBuilder(pipeline.graph_config)

        candidates, metadata = pipeline._graph_candidates("fallback question")

        self.assertEqual(candidates, [(c1, 0.7)])
        self.assertEqual(pipeline.retriever.top_k, 9)
        self.assertEqual(metadata["graph_route"], "flat_fallback")

    def test_cross_encoder_reranker_uses_lexical_fallback_when_disabled(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "dense retrieval background")
        c2 = Chunk("c2", "d1", "Paper", "method", "keyword reranker evidence")
        reranker = CrossEncoderReranker(model_name=None)

        ranked = reranker.rerank("Which chunk has keyword evidence?", [(c1, 0.1), (c2, 0.1)], top_k=1)

        self.assertEqual(ranked[0][0].chunk_id, "c2")

    def test_semantic_reranker_pipeline_reranks_dense_candidates(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "dense retrieval background")
        c2 = Chunk("c2", "d1", "Paper", "method", "keyword reranker evidence")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return [(c1, 0.9), (c2, 0.8)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.question = question
                self.candidates = candidates
                self.top_k = top_k
                return [(c2, 2.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "reranked answer"

        pipeline = SemanticRerankerPipeline.__new__(SemanticRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1

        result = pipeline.answer("Which chunk has keyword evidence?")

        self.assertEqual(result["answer"], "reranked answer")
        self.assertEqual(result["contexts"], [c2])
        self.assertEqual(result["scores"], [2.0])
        self.assertEqual(pipeline.retriever.top_k, 20)
        self.assertEqual(pipeline.reranker.top_k, 1)

    def test_wide_late_pipeline_reorders_context_and_passes_tail_reminder(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "unrelated translation text")
        c2 = Chunk("c2", "d1", "Paper", "method", "dataset benchmark evaluation")
        c3 = Chunk("c3", "d1", "Paper", "results", "Qasper dataset benchmark used for question answering.")

        class FakeRetriever:
            model_name = "fake-retriever"
            query_prefix = ""
            passage_prefix = ""
            late_chunking_backend = "late_chunking"
            late_chunking_fallback_count = 0
            late_chunking_window_count = 1
            load_error = None

            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return [(c1, 0.3), (c2, 0.8), (c3, 0.9)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                return [(chunk, score + 1.0) for chunk, score in candidates[:top_k]]

        class FakeGenerator:
            prompt_mode = "direct"
            max_input_tokens = 4096
            max_new_tokens = 96
            num_beams = 1

            def answer(self, question, contexts, *, tail_reminder_sentences=None):
                self.question = question
                self.contexts = contexts
                self.tail_reminder_sentences = tail_reminder_sentences
                return "Qasper"

        pipeline = WideLateChunkingSemanticRerankerPipeline.__new__(WideLateChunkingSemanticRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 30
        pipeline.top_k = 3
        pipeline.query_prefix = ""
        pipeline.passage_prefix = ""
        pipeline.prompt_mode = "direct"
        pipeline.context_order = "u_tail"
        pipeline.tail_reminder = True
        pipeline.min_words = 120
        pipeline.max_words = 420
        pipeline.breakpoint_threshold = 0.45
        pipeline.overlap_sentences = 2
        pipeline.late_max_tokens = 512
        pipeline.late_stride = 128

        result = pipeline.answer("Which dataset benchmark is used?")

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["contexts"][-1].chunk_id, "c3")
        self.assertEqual(pipeline.generator.tail_reminder_sentences[0], "Qasper dataset benchmark used for question answering.")
        self.assertEqual(result["chunking_mode"], "wide_semantic_late")
        self.assertEqual(result["late_chunking_backend"], "late_chunking")

    def test_semantic_hybrid_reranker_pipeline_fuses_before_reranking(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "dense-only candidate")
        c2 = Chunk("c2", "d1", "Paper", "method", "keyword bm25 candidate")

        class FakeRetriever:
            def __init__(self, results):
                self.results = results

            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return self.results

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.question = question
                self.candidates = candidates
                self.top_k = top_k
                return [(c2, 3.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "hybrid reranked answer"

        pipeline = SemanticHybridRerankerPipeline.__new__(SemanticHybridRerankerPipeline)
        pipeline.dense = FakeRetriever([(c1, 0.9), (c2, 0.2)])
        pipeline.sparse = FakeRetriever([(c2, 2.0)])
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1

        result = pipeline.answer("Which chunk has keyword evidence?")

        self.assertEqual(result["answer"], "hybrid reranked answer")
        self.assertEqual(result["contexts"], [c2])
        self.assertEqual(pipeline.dense.top_k, 20)
        self.assertEqual(pipeline.sparse.top_k, 20)
        self.assertEqual(pipeline.reranker.top_k, 1)
        self.assertEqual({chunk.chunk_id for chunk, _score in pipeline.reranker.candidates}, {"c1", "c2"})
        self.assertEqual(result["candidate_retrieval"], "semantic_dense_bm25_rrf")

    def test_contextualize_chunk_keeps_original_identity_with_enriched_text(self):
        chunk = Chunk("c1", "d1", "Paper Title", "method", "The model uses reranking.")

        contextual = contextualize_chunk(chunk, document_summary="Paper Title. Abstract summary.")

        self.assertEqual(contextual.chunk_id, chunk.chunk_id)
        self.assertIn("Document: Paper Title", contextual.text)
        self.assertIn("Section: method", contextual.text)
        self.assertIn("Summary: Paper Title. Abstract summary.", contextual.text)
        self.assertIn(chunk.text, contextual.text)

    def test_short_document_summary_uses_title_and_abstract_sentences(self):
        record = {
            "title": "A Paper",
            "abstract": "First abstract sentence. Second abstract sentence. Third abstract sentence.",
        }

        summary = short_document_summary(record)

        self.assertIn("A Paper", summary)
        self.assertIn("First abstract sentence.", summary)
        self.assertIn("Second abstract sentence.", summary)
        self.assertNotIn("Third abstract sentence.", summary)

    def test_contextual_semantic_reranker_returns_original_contexts_to_generator(self):
        original = Chunk("c1", "d1", "Paper", "method", "Original evidence text.")
        contextual = Chunk("c1", "d1", "Paper", "method", "Summary text.\n\nOriginal evidence text.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.top_k = top_k
                return [(contextual, 0.8)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.candidates = candidates
                self.top_k = top_k
                return [(contextual, 2.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.contexts = contexts
                return "answer"

        pipeline = ContextualSemanticRerankerPipeline.__new__(ContextualSemanticRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1
        pipeline.original_by_id = {"c1": original}
        pipeline.document_summary = "Summary text."

        result = pipeline.answer("What is the evidence?")

        self.assertEqual(result["contexts"], [original])
        self.assertEqual(pipeline.generator.contexts, [original])
        self.assertEqual(pipeline.reranker.candidates[0][0].text, contextual.text)
        self.assertEqual(result["retrieval_context_mode"], "contextualized_embed_rerank_original_generate")

    def test_qwen_direct_pipeline_generates_without_filter_parser(self):
        c1 = Chunk("c1", "d1", "Paper", "method", "The dataset used is Qasper.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.top_k = top_k
                return [(c1, 0.8)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.top_k = top_k
                return [(c1, 2.0)]

        class FakeQwenGenerator:
            model_name = "fake-qwen"
            max_input_tokens = 123
            max_new_tokens = 45

            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "Qasper"

        pipeline = SemanticRerankerQwenDirectPipeline.__new__(SemanticRerankerQwenDirectPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeQwenGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["contexts"], [c1])
        self.assertEqual(result["generator_family"], "qwen_direct")
        self.assertEqual(result["generator_model"], "fake-qwen")
        self.assertNotIn("filter_route", result)

    def test_oracle_gold_context_uses_evidence_and_generator(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset is used?",
            gold_answers=["Qasper"],
            evidence=["The dataset used is Qasper."],
        )

        class FakeGenerator:
            def answer(self, question, contexts):
                self.contexts = contexts
                return "Qasper"

        pipeline = OracleGoldContextPipeline.__new__(OracleGoldContextPipeline)
        pipeline.generator = FakeGenerator()

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["contexts"][0].section, "oracle_gold_evidence")
        self.assertEqual(result["oracle_context_source"], "gold_evidence")
        self.assertEqual(result["route"], "generate")

    def test_oracle_gold_context_default_does_not_pass_long_input_kwargs(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset is used?",
            gold_answers=["Qasper"],
            evidence=["The dataset used is Qasper."],
        )

        class FakeGenerator:
            def answer(self, question, contexts, **kwargs):
                self.kwargs = kwargs
                return "Qasper"

        pipeline = OracleGoldContextPipeline.__new__(OracleGoldContextPipeline)
        pipeline.generator = FakeGenerator()

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(pipeline.generator.kwargs, {})

    def test_oracle_gold_context_longinput_passes_max_input_tokens(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset is used?",
            gold_answers=["Qasper"],
            evidence=["The dataset used is Qasper."],
        )

        class FakeGenerator:
            def answer(self, question, contexts, **kwargs):
                self.kwargs = kwargs
                return "Qasper"

        pipeline = OracleGoldContextPipeline.__new__(OracleGoldContextPipeline)
        pipeline.generator = FakeGenerator()
        pipeline.max_input_tokens = 4096

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(pipeline.generator.kwargs, {"max_input_tokens": 4096})
        self.assertEqual(result["max_input_tokens"], 4096)

    def test_oracle_gold_context_falls_back_to_gold_answer_text(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset is used?",
            gold_answers=["Qasper"],
            evidence=[],
        )

        class FakeGenerator:
            def answer(self, question, contexts):
                return contexts[0].text

        pipeline = OracleGoldContextPipeline.__new__(OracleGoldContextPipeline)
        pipeline.generator = FakeGenerator()

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["contexts"][0].section, "oracle_gold_answer_text")
        self.assertEqual(result["oracle_context_source"], "gold_answer_text")

    def test_oracle_gold_context_abstains_without_usable_gold(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset is used?",
            gold_answers=["Unanswerable"],
            evidence=[],
        )
        pipeline = OracleGoldContextPipeline.__new__(OracleGoldContextPipeline)

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Unanswerable")
        self.assertEqual(result["route"], "abstain")
        self.assertEqual(result["oracle_context_source"], "none")

    def test_oracle_u_tail_places_best_last_and_second_best_first(self):
        chunks = [
            Chunk("c1", "d1", "Paper", "s", "unrelated neural translation text"),
            Chunk("c2", "d1", "Paper", "s", "dataset discussion"),
            Chunk("c3", "d1", "Paper", "s", "dataset qasper benchmark evaluation"),
            Chunk("c4", "d1", "Paper", "s", "qasper dataset benchmark used for question answering"),
        ]

        ordered = oracle_u_tail_reorder("Which dataset benchmark is used?", chunks)

        self.assertEqual(ordered[0].chunk_id, "c3")
        self.assertEqual(ordered[-1].chunk_id, "c4")

    def test_oracle_context_score_uses_question_and_context_only(self):
        high_question_overlap = Chunk("c1", "d1", "Paper", "s", "dataset benchmark evaluation")
        answer_only = Chunk("c2", "d1", "Paper", "s", "Qasper")

        high_score = oracle_question_overlap_score("Which dataset benchmark is used?", high_question_overlap)
        answer_only_score = oracle_question_overlap_score("Which dataset benchmark is used?", answer_only)

        self.assertGreater(high_score, answer_only_score)

    def test_oracle_prompt_budget_caps_contexts_and_records_drops(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset benchmark is used?",
            gold_answers=["Qasper"],
            evidence=[
                "unrelated text",
                "dataset benchmark evaluation",
                "qasper dataset benchmark",
                "benchmark dataset question answering",
                "dataset benchmark method",
                "dataset benchmark results",
            ],
        )

        class FakeGenerator:
            max_input_tokens = 4096
            max_new_tokens = 96
            num_beams = 1

            def answer(self, question, contexts, *, tail_reminder_sentences=None):
                self.contexts = contexts
                self.tail_reminder_sentences = tail_reminder_sentences
                return "Qasper"

        pipeline = OracleGoldContextPromptAblationPipeline.__new__(OracleGoldContextPromptAblationPipeline)
        pipeline.generator = FakeGenerator()
        pipeline.prompt_mode = "direct"
        pipeline.context_order = "u_tail"
        pipeline.context_budget = 5
        pipeline.tail_reminder = False

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(len(result["contexts"]), 5)
        self.assertEqual(result["oracle_context_original_count"], 6)
        self.assertEqual(result["oracle_context_dropped"], 1)
        self.assertEqual(result["oracle_context_count"], 5)

    def test_oracle_prompt_builder_has_delimiters_repeated_instruction_and_no_citation_format(self):
        context = Chunk("c1", "d1", "Paper", "results", "The dataset used is Qasper.")

        prompt = OraclePromptSeq2SeqGenerator.build_prompt(
            "Which dataset is used?",
            [context],
            prompt_mode="extractive",
            tail_reminder_sentences=["The dataset used is Qasper."],
        )

        self.assertIn("<EVIDENCE>", prompt)
        self.assertIn("</EVIDENCE>", prompt)
        self.assertIn("Final instruction", prompt)
        self.assertIn("ANSWER_CRITICAL_EVIDENCE", prompt)
        self.assertNotIn("[chunk_id]", prompt)
        self.assertNotIn("Citations are MANDATORY", prompt)

    def test_oracle_tail_reminder_selects_question_overlap_sentences(self):
        contexts = [
            Chunk("c1", "d1", "Paper", "s", "The model is trained on Common Crawl. The dataset used is Qasper."),
            Chunk("c2", "d1", "Paper", "s", "The method improves reranking."),
        ]

        reminders = oracle_tail_reminder_sentences("Which dataset is used?", contexts, limit=1)

        self.assertEqual(reminders, ["The dataset used is Qasper."])

    def test_oracle_prompt_pipeline_passes_tail_reminder_and_generation_metadata(self):
        example = QAExample(
            doc_id="d1",
            question_id="q1",
            title="Paper",
            question="Which dataset is used?",
            gold_answers=["Qasper"],
            evidence=["The dataset used is Qasper."],
        )

        class FakeGenerator:
            max_input_tokens = 4096
            max_new_tokens = 96
            num_beams = 4

            def answer(self, question, contexts, *, tail_reminder_sentences=None):
                self.tail_reminder_sentences = tail_reminder_sentences
                return "Qasper"

        pipeline = OracleGoldContextPromptAblationPipeline.__new__(OracleGoldContextPromptAblationPipeline)
        pipeline.generator = FakeGenerator()
        pipeline.prompt_mode = "direct"
        pipeline.context_order = "u_tail"
        pipeline.context_budget = None
        pipeline.tail_reminder = True

        result = pipeline.answer_example(example)

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["num_beams"], 4)
        self.assertEqual(result["max_input_tokens"], 4096)
        self.assertEqual(result["tail_reminder_sentence_count"], 1)
        self.assertEqual(pipeline.generator.tail_reminder_sentences, ["The dataset used is Qasper."])

    def test_sufficient_context_gate_accepts_clear_answer_context(self):
        gate = SufficientContextGate()
        context = Chunk("c1", "d1", "Paper", "results", "The dataset used is Qasper for scientific paper question answering.")

        decision = gate.decide("Which dataset is used?", [context])

        self.assertTrue(decision.sufficient)
        self.assertIn("dataset", decision.matched_evidence_terms)

    def test_sufficient_context_gate_rejects_unrelated_context(self):
        gate = SufficientContextGate()
        context = Chunk("c1", "d1", "Paper", "intro", "The paper studies neural machine translation.")

        decision = gate.decide("Which dataset is used?", [context])

        self.assertFalse(decision.sufficient)

    def test_sufficient_context_gate_rejects_overlap_without_answer_signal(self):
        gate = SufficientContextGate()
        context = Chunk("c1", "d1", "Paper", "related", "Dataset selection and usage appear in the related work discussion.")

        decision = gate.decide("Which dataset is used?", [context])

        self.assertFalse(decision.sufficient)

    def test_self_route_pipeline_abstains_without_calling_generator(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "The paper studies neural machine translation.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.top_k = top_k
                return [(c1, 0.4)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                return candidates[:top_k]

        class FakeGenerator:
            called = False

            def answer(self, question, contexts):
                self.called = True
                return "should not be called"

        pipeline = SelfRouteSemanticRerankerPipeline.__new__(SelfRouteSemanticRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.gate = SufficientContextGate()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1
        pipeline.query_prefix = ""
        pipeline.passage_prefix = ""

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Unanswerable")
        self.assertEqual(result["route"], "abstain")
        self.assertFalse(result["sufficient"])
        self.assertFalse(pipeline.generator.called)

    def test_self_route_pipeline_generates_when_context_is_sufficient(self):
        c1 = Chunk("c1", "d1", "Paper", "results", "The dataset used is Qasper for scientific paper question answering.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.top_k = top_k
                return [(c1, 0.8)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.top_k = top_k
                return [(c1, 2.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "Qasper"

        pipeline = SelfRouteSemanticRerankerPipeline.__new__(SelfRouteSemanticRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.gate = SufficientContextGate()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1
        pipeline.query_prefix = ""
        pipeline.passage_prefix = ""

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["contexts"], [c1])
        self.assertEqual(result["scores"], [2.0])
        self.assertEqual(result["route"], "generate")
        self.assertTrue(result["sufficient"])
        self.assertIn("sufficient_confidence", result)
        self.assertIn("sufficient_reason", result)

    def test_qwen_filter_parse_generate_output(self):
        raw = '{"route":"generate","selected_indices":[1, "2", 99],"evidence_pack":"Qasper is used.","reason":"direct evidence"}'

        decision = QwenEvidenceFilterCompressor.parse_output(raw, context_count=2)

        self.assertEqual(decision.route, "generate")
        self.assertEqual(decision.selected_indices, [1, 2])
        self.assertEqual(decision.evidence_pack, "Qasper is used.")
        self.assertIsNone(decision.parse_error)

    def test_qwen_filter_parse_malformed_output_abstains(self):
        decision = QwenEvidenceFilterCompressor.parse_output("not json", context_count=2)

        self.assertEqual(decision.route, "abstain")
        self.assertEqual(decision.selected_indices, [])
        self.assertEqual(decision.reason, "filter_parse_error")
        self.assertIsNotNone(decision.parse_error)

    def test_e5_qwen_filter_pipeline_abstains_without_calling_generator(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "Unrelated background.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.top_k = top_k
                return [(c1, 0.7)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.top_k = top_k
                return candidates

        class FakeFilter:
            model_name = "fake-qwen"

            def filter(self, question, contexts):
                return QwenEvidenceFilterCompressor.parse_output(
                    '{"route":"abstain","selected_indices":[],"evidence_pack":"","reason":"insufficient evidence"}',
                    context_count=len(contexts),
                )

        class FakeGenerator:
            called = False

            def answer(self, question, contexts):
                self.called = True
                return "should not run"

        pipeline = E5QwenFilterGeneratorPipeline.__new__(E5QwenFilterGeneratorPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.filter_compressor = FakeFilter()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 30
        pipeline.filter_top_k = 8
        pipeline.filter_mode = "hard_route"
        pipeline.answer_with_qwen = False
        pipeline.query_prefix = "query: "
        pipeline.passage_prefix = "passage: "

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Unanswerable")
        self.assertEqual(result["filter_route"], "abstain")
        self.assertFalse(pipeline.generator.called)
        self.assertEqual(result["filter_model"], "fake-qwen")

    def test_e5_qwen_filter_pipeline_generates_from_evidence_pack(self):
        c1 = Chunk("c1", "d1", "Paper", "method", "The dataset used is Qasper.")
        c2 = Chunk("c2", "d1", "Paper", "results", "The model reports strong QA results.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.top_k = top_k
                return [(c1, 0.8), (c2, 0.7)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.top_k = top_k
                return [(c1, 3.0), (c2, 2.0)]

        class FakeFilter:
            model_name = "fake-qwen"

            def filter(self, question, contexts):
                self.contexts = contexts
                return QwenEvidenceFilterCompressor.parse_output(
                    '{"route":"generate","selected_indices":[1],"evidence_pack":"The dataset used is Qasper.","reason":"direct evidence"}',
                    context_count=len(contexts),
                )

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "Qasper"

        pipeline = E5QwenFilterGeneratorPipeline.__new__(E5QwenFilterGeneratorPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.filter_compressor = FakeFilter()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 30
        pipeline.filter_top_k = 8
        pipeline.filter_mode = "hard_route"
        pipeline.answer_with_qwen = False
        pipeline.query_prefix = "query: "
        pipeline.passage_prefix = "passage: "

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Qasper")
        self.assertEqual(result["contexts"], [c1])
        self.assertEqual(result["scores"], [3.0])
        self.assertEqual(result["filter_route"], "generate")
        self.assertEqual(result["selected_context_indices"], [1])
        self.assertEqual(pipeline.generator.contexts[0].section, "qwen_evidence_pack")
        self.assertEqual(pipeline.generator.contexts[0].text, "The dataset used is Qasper.")

    def test_e5_qwen_answer_only_uses_qwen_pack_as_final_answer(self):
        c1 = Chunk("c1", "d1", "Paper", "method", "The dataset used is Qasper.")

        class FakeRetriever:
            def search(self, question, *, top_k):
                return [(c1, 0.8)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                return [(c1, 3.0)]

        class FakeFilter:
            model_name = "fake-qwen"

            def filter(self, question, contexts):
                return QwenEvidenceFilterCompressor.parse_output(
                    '{"route":"generate","selected_indices":[1],"evidence_pack":"Qasper","reason":"final answer"}',
                    context_count=len(contexts),
                )

        class FakeGenerator:
            called = False

            def answer(self, question, contexts):
                self.called = True
                return "should not run"

        pipeline = E5QwenFilterGeneratorPipeline.__new__(E5QwenFilterGeneratorPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.filter_compressor = FakeFilter()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 30
        pipeline.filter_top_k = 8
        pipeline.filter_mode = "answer_only"
        pipeline.answer_with_qwen = True
        pipeline.query_prefix = "query: "
        pipeline.passage_prefix = "passage: "

        result = pipeline.answer("Which dataset is used?")

        self.assertEqual(result["answer"], "Qasper")
        self.assertFalse(pipeline.generator.called)
        self.assertEqual(result["filter_mode"], "answer_only")
        self.assertTrue(result["answer_with_qwen"])

    def test_raptor_tree_builder_creates_parent_nodes(self):
        leaves = [Chunk(f"c{i}", "d1", "Paper", "s", f"Sentence {i}. More evidence.") for i in range(5)]
        builder = RaptorTreeBuilder(RaptorConfig(group_size=2, max_levels=1), embedder=None)

        parents = builder.build(leaves)

        self.assertGreaterEqual(len(parents), 2)
        self.assertTrue(all("raptor_level_1" == parent.section for parent in parents))

    def test_raptor_tree_builder_uses_injected_summarizer(self):
        class FakeSummarizer:
            def summarize(self, chunks):
                return f"summary of {len(chunks)} chunks"

        leaves = [Chunk(f"c{i}", "d1", "Paper", "s", f"Sentence {i}.") for i in range(2)]
        builder = RaptorTreeBuilder(
            RaptorConfig(group_size=2, max_levels=1),
            embedder=None,
            summarizer=FakeSummarizer(),
        )

        parents = builder.build(leaves)

        self.assertEqual(parents[0].text, "summary of 2 chunks")

    def test_gmm_and_agglomerative_builders_create_parent_nodes(self):
        leaves = [Chunk(f"c{i}", "d1", "Paper", "s", f"Sentence {i}.") for i in range(6)]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]

        gmm_builder = GMMRaptorTreeBuilder(
            RaptorConfig(group_size=2, max_levels=1, random_state=7),
            embedder=FakeEmbedder(vectors),
        )
        agglomerative_builder = AgglomerativeRaptorTreeBuilder(
            RaptorConfig(group_size=2, max_levels=1),
            embedder=FakeEmbedder(vectors),
        )

        self.assertGreaterEqual(len(gmm_builder.build(leaves)), 1)
        self.assertEqual(gmm_builder.last_backend, "umap_gmm_soft")
        self.assertGreaterEqual(len(agglomerative_builder.build(leaves)), 2)
        self.assertEqual(agglomerative_builder.last_backend, "agglomerative_dendrogram_n3_n6_root")

    def test_semantic_raptor_leiden_reranker_reranks_collapsed_candidates(self):
        c1 = Chunk("leaf", "d1", "Paper", "method", "leaf evidence")
        c2 = Chunk("parent", "d1", "Paper", "raptor_level_1", "parent summary evidence")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return [(c1, 0.7), (c2, 0.6)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.question = question
                self.candidates = candidates
                self.top_k = top_k
                return [(c2, 4.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "raptor reranked answer"

        pipeline = SemanticRaptorLeidenRerankerPipeline.__new__(SemanticRaptorLeidenRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1
        pipeline.parent_count = 3
        pipeline.raptor_backend = "leiden"

        result = pipeline.answer("Which evidence?")

        self.assertEqual(result["answer"], "raptor reranked answer")
        self.assertEqual(result["contexts"], [c2])
        self.assertEqual(result["scores"], [4.0])
        self.assertEqual(result["raptor_parent_count"], 3)
        self.assertEqual(result["raptor_backend"], "leiden")
        self.assertEqual(pipeline.retriever.top_k, 20)
        self.assertEqual(pipeline.reranker.top_k, 1)

if __name__ == "__main__":
    unittest.main()
