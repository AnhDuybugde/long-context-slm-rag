# Base RAG Scope

Last update: 2026-05-19

## Why This Baseline Exists

The first phase needs a stable reference point before adding advanced RAG methods. This version is intentionally simple so later improvements can be compared fairly.

## Dataset

- Primary dataset: Qasper
- Source: https://allenai.org/data/qasper
- Loader used in code: `datasets.load_dataset("allenai/qasper")`

Qasper is a good first benchmark because it focuses on question answering over scientific papers, which matches the long-document research direction better than short open-domain QA datasets.

## Included In The Base Version

- Qasper dataset loading.
- Per-document indexing.
- Fixed-size overlapping word chunking.
- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`.
- Top-k context selection.
- Small sequence-to-sequence generator with `google/flan-t5-base`.
- Grounded prompt that asks the model to answer only from context or return `Unanswerable`.
- Lightweight evaluation:
  - context precision
  - context recall
  - faithfulness
  - answer relevancy
  - token-level F1 against gold answers
  - weak answer-string recall in retrieved contexts
- JSONL prediction export for later analysis.
- Summary JSON export with config and aggregate metrics.
- Colab/Kaggle notebook launcher in `notebooks/`.

## Not Included Yet

These are intentionally excluded from the baseline and should be added one by one in later experiments:

- Semantic chunking.
- BM25 or hybrid retrieval.
- RRF/interleaving fusion.
- Cross-encoder reranking.
- RAPTOR tree indexing.
- Context compression.
- U-shaped context reordering.
- SELF-ROUTE or long-context fallback.
- RAGAS evaluation.
- Domain-specific environmental corpus.
- Fine-tuned embedding model or fine-tuned SLM.

## Suggested Experiment Order

1. Run the full baseline split and save metrics.
2. Replace fixed chunking with semantic chunking.
3. Add BM25 and compare sparse vs dense vs hybrid.
4. Add cross-encoder reranking.
5. Add context reordering.
6. Add RAPTOR.
7. Add SELF-ROUTE fallback.
8. Move from Qasper to custom environmental data.

## Development Workflow

1. Implement or update `.py` source modules first.
2. Run tests for the changed module.
3. If tests fail, fix the source and rerun tests.
4. If tests pass, continue to the next implementation step.
5. Only after the `.py` workflow is stable, summarize the working code into `.ipynb` for Colab/Kaggle.

The notebook should be an experiment runner, not the primary source of truth.

## Base Training Meaning

The baseline does not fine-tune model weights. "Training" means fitting the RAG
runtime for an experiment: chunking each Qasper paper, embedding chunks, building
the per-document dense index, generating answers with the selected SLM, and
writing comparable evaluation artifacts.

By default the CLI should run the full selected split. Use `--limit` only for
debug/smoke tests.
