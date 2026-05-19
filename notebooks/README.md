# Notebooks

Use this folder for Colab/Kaggle experiments.

Current notebook flow:

1. Install `datasets` and `pyarrow`.
2. Load Qasper from its Hugging Face Parquet exports with `datasets.load_dataset("parquet", data_files=...)`.
3. Inspect dataset splits, schema, sample document text, and QA pairs.
4. Export a small validation preview JSONL.
5. Optionally run the base RAG training/evaluation runner after the repo code is available in the Kaggle/Colab working directory.

The dataset exploration cells do not require cloning GitHub. The base RAG runner section needs the repository source code because the `.py` modules are the source of truth.
