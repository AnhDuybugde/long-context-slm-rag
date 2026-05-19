# Notebooks

Use this folder for Colab/Kaggle experiments.

Current notebook flow:

1. Install `datasets` and `pyarrow`.
2. Load Qasper from its Hugging Face Parquet exports with `datasets.load_dataset("parquet", data_files=...)`.
3. Inspect dataset splits, schema, sample document text, and QA pairs.
4. Export a small validation preview JSONL.
5. Optionally run the self-contained base RAG training/evaluation runner.

The Kaggle notebook is self-contained because importing local repo modules can be awkward there. The `.py` modules remain the source of truth for tests and maintainability.
