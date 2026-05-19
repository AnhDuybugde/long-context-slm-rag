# Notebooks

Use this folder for Colab/Kaggle experiments.

Current notebook flow:

1. Load Qasper directly with `datasets.load_dataset("allenai/qasper")`.
2. Inspect dataset splits, schema, sample document text, and QA pairs.
3. Export a small validation preview JSONL.

This notebook does not require cloning GitHub. It is only for verifying the dataset on Kaggle/Colab. The full RAG runner can be added after the dataset loading step is stable.
