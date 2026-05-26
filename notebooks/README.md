# Notebooks

Use this folder for Colab/Kaggle experiments.

Current notebook flow:

1. Install `datasets` and `pyarrow`.
2. Load Qasper from its Hugging Face Parquet exports with `datasets.load_dataset("parquet", data_files=...)`.
3. Inspect dataset splits, schema, sample document text, and QA pairs.
4. Export a small validation preview JSONL.
5. Optionally run the self-contained base RAG training/evaluation runner.

The Kaggle notebook is self-contained because importing local repo modules can be awkward there. The `.py` modules remain the source of truth for tests and maintainability.

## Improved Notebook

`qasper_improved_rag_colab.ipynb` is a separate self-contained notebook with the first improvement package:

- Dense retrieval + BM25 sparse retrieval.
- Reciprocal Rank Fusion.
- U-shaped context reordering.
- A stricter evidence-focused generation prompt.

Run it after the baseline notebook and compare `outputs/improved_rag_qasper_summary.json` with `outputs/base_rag_qasper_summary.json`.

## Independent Variant Notebooks

`notebooks/independent_variants/` contains Kaggle/Colab standalone notebooks for
running one method at a time without importing repo files:

- `qasper_long_context_survey_standalone.ipynb`
- `qasper_base_dense_standalone.ipynb`
- `qasper_bm25_only_standalone.ipynb`
- `qasper_dense_u_shape_standalone.ipynb`
- `qasper_dense_recency_heavy_standalone.ipynb`
- `qasper_hybrid_rrf_standalone.ipynb`
- `qasper_semantic_chunking_dense_standalone.ipynb`
- `qasper_semantic_chunking_reranker_standalone.ipynb`
- `qasper_semantic_chunking_reranker_ablation_batch_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_baseline_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_strict_prompt_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_extractive_prompt_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_citation_prompt_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_neighbor1_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_wide_latechunk_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_wide_latechunk_sentence_select_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_wide_latechunk_high_recall_compress_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_wide_latechunk_graphrag_raptor_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_wide_latechunk_graphrag_raptor_sentence_select_standalone.ipynb`
- `advanced/qasper_sem_rerank_minilm_wide_latechunk_graphrag_raptor_high_recall_compress_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_base_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_base_strict_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_wide_latechunk_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_wide_latechunk_sentence_select_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_wide_latechunk_high_recall_compress_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_wide_latechunk_graphrag_raptor_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_wide_latechunk_graphrag_raptor_sentence_select_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_wide_latechunk_graphrag_raptor_high_recall_compress_standalone.ipynb`
- `advanced/qasper_sem_rerank_bge_base_standalone.ipynb`
- `advanced/qasper_sem_rerank_gte_base_standalone.ipynb`
- `advanced/qasper_sem_rerank_e5_neighbor1_strict_standalone.ipynb`
- `qasper_semantic_chunking_hybrid_reranker_standalone.ipynb`
- `qasper_dense_reranker_standalone.ipynb`
- `qasper_raptor_extractive_standalone.ipynb`
- `qasper_raptor_gmm_abstractive_standalone.ipynb`
- `qasper_raptor_leiden_abstractive_standalone.ipynb`
- `qasper_raptor_agglomerative_abstractive_standalone.ipynb`
- `qasper_semantic_raptor_leiden_reranker_standalone.ipynb`
- `qasper_self_route_minilm_abstain_standalone.ipynb`
- `qasper_self_route_e5_abstain_standalone.ipynb`
- `qasper_oracle_gold_context_flan_base_generator_boost_standalone.ipynb`
- `qasper_sem_rerank_minilm_qwen15_direct_standalone.ipynb`
- `qasper_sem_rerank_minilm_qwen05_direct_standalone.ipynb`
- `qasper_contextual_sem_rerank_minilm_flan_base_standalone.ipynb`
- `qasper_e5_qwen_filter_flan_base_standalone.ipynb`
- `qasper_e5_qwen_filter_flan_large_standalone.ipynb`
- `qasper_e5_qwen_compress_only_flan_large_standalone.ipynb`
- `qasper_e5_qwen_soft_route_flan_large_standalone.ipynb`
- `qasper_e5_qwen_answer_only_standalone.ipynb`

Use these for fair independent comparisons before choosing which method to
develop further.

The newer advanced notebooks are still isolated ablations. `raptor_extractive`
is an offline extractive proxy for RAPTOR. `raptor_leiden_abstractive` is closer
to the research idea because it uses abstractive parent summaries and installs
`igraph`/`leidenalg` for Leiden clustering. SELF-ROUTE, GeoFM, and table/SQL
router variants are intentionally excluded from the independent comparison set.
