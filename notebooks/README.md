# Notebooks

Use this folder for Colab/Kaggle experiments.

Current notebook flow:

1. Install `datasets` and `pyarrow`.
2. Load Qasper from its Hugging Face Parquet exports with `datasets.load_dataset("parquet", data_files=...)`.
3. Inspect dataset splits, schema, sample document text, and QA pairs.
4. Export a small validation preview JSONL.

This notebook does not require cloning GitHub. It is only for verifying the dataset on Kaggle/Colab. The full RAG runner can be added after the dataset loading step is stable.
