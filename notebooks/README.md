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
