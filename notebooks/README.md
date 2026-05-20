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
- `qasper_dense_reranker_standalone.ipynb`
- `qasper_raptor_extractive_standalone.ipynb`
- `qasper_raptor_leiden_abstractive_standalone.ipynb`

Use these for fair independent comparisons before choosing which method to
develop further.

The newer advanced notebooks are still isolated ablations. `raptor_extractive`
is an offline extractive proxy for RAPTOR. `raptor_leiden_abstractive` is closer
to the research idea because it uses abstractive parent summaries and installs
`igraph`/`leidenalg` for Leiden clustering. SELF-ROUTE, GeoFM, and table/SQL
router variants are intentionally excluded from the independent comparison set.
