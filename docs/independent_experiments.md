# Independent Long-Context Experiments

Goal: run one method at a time on Qasper long-document QA, compare metrics and runtime, then choose the method worth developing further.

## 1. Survey Long-Context Coverage

```bash
.venv\Scripts\python.exe -m src.qasper_base_rag.survey_long_context --split validation
```

Current validation result:

| Threshold | Documents | QA examples | Meaning |
| --- | ---: | ---: | --- |
| >= 1k words | 280 / 281 | 1003 / 1005 | Almost full validation is non-short document QA. |
| >= 3k words | 165 / 281 | 583 / 1005 | Good default long-document subset. |
| >= 5k words | 29 / 281 | 94 / 1005 | Small but useful stricter subset. |
| >= 8k words | 9 / 281 | 30 / 1005 | Too small for main benchmark. |
| >= 12k words | 4 / 281 | 14 / 1005 | Case-study only. |

Recommended reporting:

- Full validation: comparable with earlier notebook outputs.
- `--min-doc-words 3000`: long-context-focused subset.
- Optional `--min-doc-words 5000`: stricter stress test, but low sample size.

## 2. Run One Independent Variant

Use `evaluate_experiment.py`; run one `--variant` per command.

```bash
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant base_dense --split validation --min-doc-words 3000
```

Available source-code variants:

| Variant | What It Tests |
| --- | --- |
| `base_dense` | Baseline dense retrieval, score-order context. |
| `bm25_only` | Sparse keyword retrieval only. |
| `dense_u_shape` | Dense retrieval plus U-shaped context reordering. |
| `dense_recency_heavy` | Dense retrieval plus strongest evidence at the end of the prompt. |
| `hybrid_rrf` | Dense + BM25 with reciprocal rank fusion; included as a comparison variant, not the current target. |
| `semantic_chunking_dense` | Semantic sentence-boundary chunking plus dense retrieval. |
| `dense_reranker` | Dense retrieve-then-rerank with a cross-encoder; falls back to lexical reranking if the reranker cannot load. |
| `raptor_extractive` | RAPTOR-style collapsed tree with extractive parent summaries and graph-similarity grouping; proxy for full RAPTOR. |
| `raptor_leiden_abstractive` | More faithful RAPTOR with recursive abstractive summaries and Leiden graph clustering when dependencies install. |

Example batch:

```bash
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant base_dense --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant bm25_only --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant dense_u_shape --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant dense_recency_heavy --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant hybrid_rrf --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant semantic_chunking_dense --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant dense_reranker --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant raptor_extractive --split validation --min-doc-words 3000
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant raptor_leiden_abstractive --split validation --min-doc-words 3000
```

For a smoke test:

```bash
.venv\Scripts\python.exe -m src.qasper_base_rag.evaluate_experiment --variant dense_recency_heavy --split validation --min-doc-words 3000 --limit 10
```

Outputs are written to `outputs/experiments/`.

## 3. Kaggle Standalone Notebooks

Use these when Kaggle cannot import local repo modules. Each notebook contains
the full code it needs and should be uploaded/run independently.

| Notebook | Variant |
| --- | --- |
| `notebooks/independent_variants/qasper_long_context_survey_standalone.ipynb` | Survey Qasper document lengths and long-context coverage. |
| `notebooks/independent_variants/qasper_base_dense_standalone.ipynb` | `base_dense` |
| `notebooks/independent_variants/qasper_bm25_only_standalone.ipynb` | `bm25_only` |
| `notebooks/independent_variants/qasper_dense_u_shape_standalone.ipynb` | `dense_u_shape` |
| `notebooks/independent_variants/qasper_dense_recency_heavy_standalone.ipynb` | `dense_recency_heavy` |
| `notebooks/independent_variants/qasper_hybrid_rrf_standalone.ipynb` | `hybrid_rrf` |
| `notebooks/independent_variants/qasper_semantic_chunking_dense_standalone.ipynb` | `semantic_chunking_dense` |
| `notebooks/independent_variants/qasper_dense_reranker_standalone.ipynb` | `dense_reranker` |
| `notebooks/independent_variants/qasper_raptor_extractive_standalone.ipynb` | `raptor_extractive` |
| `notebooks/independent_variants/qasper_raptor_leiden_abstractive_standalone.ipynb` | `raptor_leiden_abstractive` |

Notebook defaults:

- `SPLIT = "validation"`
- `MIN_DOC_WORDS = 3000`
- `LIMIT = None`
- outputs go to `outputs/independent/`

For a quick smoke test on Kaggle, change `LIMIT = None` to `LIMIT = 10`.
