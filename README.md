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
python -m src.qasper_base_rag.evaluate --split validation
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

## Base RAG Training/Evaluation

In this baseline, "training" means building the per-document RAG index and running
the SLM generation/evaluation loop. No model weights are fine-tuned yet.

```bash
python -m src.qasper_base_rag.evaluate \
  --split validation \
  --top-k 5 \
  --output-predictions outputs/base_rag_qasper_predictions.jsonl \
  --output-summary outputs/base_rag_qasper_summary.json
```

For a quick smoke test, add a small limit:

```bash
python -m src.qasper_base_rag.evaluate --split validation --limit 5
```

Artifacts:

- `outputs/base_rag_qasper_predictions.jsonl`: one row per QA example.
- `outputs/base_rag_qasper_summary.json`: config and aggregate metrics.

For Colab or Kaggle, open the notebook in `notebooks/` and run the cells there.
