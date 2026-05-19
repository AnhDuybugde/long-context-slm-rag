# Long-Context SLM RAG

This project starts with a clean baseline RAG system on Qasper, then leaves room for later improvements such as semantic chunking, hybrid retrieval, reranking, RAPTOR, context reordering, and routing.

## Phase 1: Base RAG

Dataset: Qasper (`allenai/qasper`)  
Goal: build and evaluate a simple RAG pipeline with a small language model.

Base pipeline:

1. Load Qasper documents and QA pairs.
2. Split each paper into simple fixed-size overlapping chunks.
3. Embed chunks with a sentence-transformer retriever.
4. Retrieve top-k chunks for each question.
5. Generate an answer with a small Hugging Face seq2seq model.
6. Evaluate with lightweight baseline metrics.

This baseline intentionally does not include advanced methods yet. It is the reference point for future ablation studies.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.qasper_base_rag.evaluate --split validation --limit 20
```

Qasper's old Hugging Face loader uses a dataset script (`qasper.py`), which is
not supported by newer `datasets` versions. This repo loads the standard Parquet
exports directly instead of calling `load_dataset("allenai/qasper")`.

The evaluator exports per-example JSONL records and prints the four main RAG comparison metrics:

- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevancy`

For the base version these are offline heuristic metrics, not LLM-as-judge metrics. They are designed to make the baseline comparable before adding RAGAS or another judge-based evaluator.

For Colab or Kaggle, open the notebook in `notebooks/` and run the cells there.
