# Agent Memory

Last update time: 2026-05-26 Asia/Bangkok

Read this file before doing something new. Keep it compact: only durable project context, current research decisions, active artifacts, verified results, and conclusions needed for later answers.

## Core Workflow

- Think before code: understand the task and repo shape before editing.
- Test before code changes when possible; for research/planning, verify claims against local artifacts or available sources.
- Project workflow: update `.py` source first, run tests, fix failures, then mirror stable code into `.ipynb`.
- Kaggle/Colab notebooks should be self-contained because the user often cannot import local repo files there.
- Prefer OOP where it makes RAG components swappable and experiments easier to compare.
- Keep methods independent unless the user explicitly asks to combine them.
- After successful verification or important new results, update this memory in concise form.

## Repository Map

| Path | Purpose |
| --- | --- |
| `.agent/skills.md` | Local project behavior rules. |
| `.agent/memory.md` | Compact persistent project memory. |
| `src/qasper_base_rag/` | Source implementation for data, chunking, retrieval, generation, metrics, trainer, and variants. |
| `notebooks/independent_variants/` | Self-contained Kaggle notebooks for independent and follow-up variants. |
| `notebooks/output/` | Local/Kaggle output summaries and predictions. |
| `docs/` | Scope and experiment documentation. |
| `tests/` | Unit tests for source components and variants. |

## Research Direction

- Goal: build and evaluate long-context-with-SLM RAG systems.
- Standard benchmark: Qasper.
- Later target domain may include environmental/legal/technical documents, but Qasper is the current comparable benchmark.
- Current research style: run independent methods, compare metrics/runtime, inspect failure cases, then go deeper into promising methods.
- Avoid prematurely combining every strong-looking component into one pipeline.
- Report both retrieval quality and generation quality; do not treat high context recall alone as success.

## Dataset Facts

- Qasper loaded via HF Parquet exports, not dataset script, because newer `datasets` versions reject dataset scripts.
- Validation split: 281 documents, 1005 QA examples.
- Long-document validation survey:
  - Documents: 281
  - QA examples: 1005
  - Word counts: min 650, p25 2395, median 3473, mean 3638.41, p75 4337, p90 5030, p95 6393, max 14882
  - `>=3000` words: 165 docs, 583 QA examples
  - `>=5000` words: 29 docs, 94 QA examples
  - `>=8000` words: 9 docs, 30 QA examples
  - `>=12000` words: 4 docs, 14 QA examples
- Main experiment table currently uses validation with `min_doc_words=3000`, so examples are usually 583.
- Parquet schema note: `qas["answers"][i]` can be a dict of columns; normalize before iterating answer records.

## Architecture Ideas From NotebookLM

- Recommended long-context SLM RAG direction:
  `Raw documents -> semantic chunking/RAPTOR -> retrieval -> hybrid fusion -> cross-encoder reranking -> context packing/reordering/compression -> SLM generation -> optional fallback`.
- Semantic chunking is useful for preserving argument boundaries, formulas, tables, clauses, and technical evidence.
- Dense retrieval handles semantic similarity; BM25/sparse retrieval helps exact terms, IDs, standards, measurements, and abbreviations.
- RRF can fuse dense and sparse candidates.
- Cross-encoder reranking is the most defensible way to improve retrieved context quality.
- U-shaped or recency-heavy ordering was tested for positional bias but did not clearly help in current results.
- SELF-ROUTE/API fallback was excluded earlier for fairness and cost, then later reintroduced only as cheap abstention variants without external LLM fallback.
- Evaluate with Token F1, Answer Recall@5, Context Recall, Context Precision, Faithfulness, Relevancy, runtime, and sec/example.

## Baseline And Early Results

- Initial baseline scaffold created under `src/qasper_base_rag/` and `notebooks/qasper_base_rag_colab.ipynb`.
- Baseline means inference/evaluation with frozen small models, not model fine-tuning.
- Full validation early baseline result:
  - Examples: 1005
  - avg_token_f1: 0.1384
  - avg_answer_string_recall_at_5: 0.2235
  - avg_context_precision: 0.7122
  - avg_context_recall: 0.8379
  - avg_faithfulness: 0.9862
  - avg_answer_relevancy: 0.0980
- Interpretation: retrieval can find useful context, but small generator/prompt gives weak answer overlap and low relevancy.

## Active Variant Mapping

Use this mapping when answering questions about the current experiment table.

| Variant | Method name |
| --- | --- |
| `raptor_leiden_abstractive` | RAPTOR with Leiden Abstractive Summarization |
| `raptor_extractive` | RAPTOR Extractive |
| `raptor_gmm_abstractive` | RAPTOR GMM Abstractive |
| `raptor_agglomerative_abstractive` | RAPTOR Agglomerative Abstractive |
| `semantic_raptor_leiden_reranker` | Semantic RAPTOR Leiden + Cross-Encoder Reranker |
| `base_dense` | Base Dense Retrieval |
| `semantic_chunking_dense` | Dense Retrieval + Semantic Chunking |
| `dense_reranker` | Dense Retrieval + Cross-Encoder Reranker |
| `dense_recency_heavy` | Dense Retrieval + Recency-Heavy Reordering |
| `dense_u_shape` | Dense Retrieval + U-Shape Reordering |
| `bm25_only` | BM25 Retrieval Only |
| `hybrid_rrf` | Hybrid Retrieval with Reciprocal Rank Fusion |
| `semantic_chunking_reranker` | Semantic Chunking + Dense Retrieval + Cross-Encoder Reranker |
| `semantic_chunking_hybrid_reranker` | Semantic Chunking + Hybrid RRF Retrieval + Cross-Encoder Reranker |
| `sem_rerank_minilm_baseline` | Semantic Chunking + MiniLM Dense + Cross-Encoder Reranker |
| `sem_rerank_minilm_baseline_tk8` | MiniLM semantic reranker with retrieve_k=30, top_k=8 |
| `sem_rerank_minilm_baseline_rk30` | MiniLM semantic reranker with retrieve_k=30 |
| `sem_rerank_minilm_extractive_prompt` | MiniLM Semantic Reranker + Extractive Prompt |
| `sem_rerank_minilm_strict_prompt` | MiniLM Semantic Reranker + Strict Grounded Prompt |
| `sem_rerank_minilm_citation_prompt` | MiniLM Semantic Reranker + Citation Prompt |
| `sem_rerank_minilm_neighbor1` | MiniLM Semantic Reranker + Neighbor Context Expansion |
| `sem_rerank_minilm_wide_latechunk` | MiniLM Wide Semantic Chunking + Late Chunking + Reranker + Generator Boost Prompt |
| `sem_rerank_bge_base` | Semantic Chunking + BGE-base Dense + Cross-Encoder Reranker |
| `sem_rerank_gte_base` | Semantic Chunking + GTE-base Dense + Cross-Encoder Reranker |
| `sem_rerank_e5_base` | Semantic Chunking + E5-base Dense + Cross-Encoder Reranker |
| `sem_rerank_e5_base_strict` | E5-base Semantic Reranker + Strict Grounded Prompt |
| `sem_rerank_e5_wide_latechunk` | E5 Wide Semantic Chunking + Late Chunking + Reranker + Generator Boost Prompt |
| `sem_rerank_e5_neighbor1_strict` | E5-base Semantic Reranker + Neighbor Context + Strict Prompt |
| `e5_qwen_filter_flan_base` | E5 Retrieval + Cross-Encoder Reranker + Qwen Evidence Filter/Compressor + Flan-T5-base Generator |
| `e5_qwen_filter_flan_large` | E5 Retrieval + Cross-Encoder Reranker + Qwen Evidence Filter/Compressor + Flan-T5-large Generator |
| `e5_qwen_soft_route_flan_large` | E5 Retrieval + Reranker + Qwen Soft Routing/Compression + Flan-T5-large |
| `e5_qwen_compress_only_flan_large` | E5 Retrieval + Reranker + Qwen Compress-Only Evidence Pack + Flan-T5-large |
| `contextual_sem_rerank_minilm_flan_base` | Contextual Semantic Chunking + MiniLM Dense + Reranker + Flan-T5-base |
| `sem_rerank_minilm_qwen05_direct` | Semantic Chunking + MiniLM Dense + Reranker + Qwen2.5-0.5B Direct Generator |
| `sem_rerank_minilm_qwen15_direct` | Semantic Chunking + MiniLM Dense + Reranker + Qwen2.5-1.5B Direct Generator |
| `oracle_gold_context_flan_base` | Oracle Gold Context + Flan-T5-base Generator |
| `oracle_gold_context_flan_base_generator_boost` | Oracle Gold Context + Flan-T5-base Generator Boost Prompt/Context Packing |

## Current Result Table Highlights

- `oracle_gold_context_flan_base`: Token F1 0.3067, Answer Recall@5 0.3210, Context Recall 0.9460, Context Precision 0.9468, Faithfulness 0.5026, Relevancy 0.1616, runtime 184.86s.
- `sem_rerank_minilm_baseline`: Token F1 0.2551, Relevancy 0.1637, runtime 245.76s.
- `sem_rerank_minilm_extractive_prompt`: Token F1 0.2549, Relevancy 0.1600, runtime 231.94s.
- `sem_rerank_minilm_strict_prompt`: Token F1 0.2481, Relevancy 0.1636.
- `semantic_chunking_reranker`: Token F1 0.2338, Answer Recall@5 0.1967, Context Recall 0.4160, Precision 0.6000.
- `semantic_chunking_hybrid_reranker`: Token F1 0.2251, Context Recall 0.4114, Precision 0.6065.
- `semantic_chunking_dense`: Token F1 0.2256, Relevancy 0.1356, but lower context recall/precision and faithfulness.
- `dense_reranker`: best early retrieval-oriented method among simple baselines: Answer Recall@5 0.2207, Context Recall 0.4603, Context Precision 0.6518.
- `sem_rerank_e5_base_tk8`: high Context Recall 0.5454 and Precision 0.6299, but Token F1 only 0.0741.
- `e5_qwen_filter_flan_base` and `e5_qwen_filter_flan_large`: high context recall/precision and near-perfect faithfulness, but relevancy near zero; Qwen filter is too strict/noisy or destroys answer signal.
- `sem_rerank_minilm_qwen15_direct`: Token F1 0.2389, Relevancy 0.2552, runtime 879.15s.
- `sem_rerank_minilm_qwen05_direct`: Token F1 0.2157, Relevancy 0.2497, runtime 660.41s.
- RAPTOR variants improve over base dense in some F1 cases but are expensive:
  - `raptor_extractive`: F1 0.1425, runtime 3043.50s.
  - `raptor_leiden_abstractive`: F1 0.1522, runtime 1655.36s.
  - `raptor_gmm_abstractive`: F1 0.1477, runtime 1447.62s.
  - `raptor_agglomerative_abstractive`: F1 0.1782, runtime 2360.66s.
  - `semantic_raptor_leiden_reranker`: F1 0.2254, runtime 3427.14s.

## Key Interpretation

- `oracle_gold_context_flan_base` is an oracle diagnostic upper bound, not a fair deployable RAG method.
- It feeds Qasper gold evidence/context directly to `google/flan-t5-base`, bypassing normal retrieval.
- Its Context Recall and Context Precision are high because the pipeline is given the answer-bearing evidence instead of retrieving it.
- Despite near-gold context, Token F1 is only 0.3067 and Relevancy 0.1616, proving that generator capacity/prompting is also a major bottleneck.
- Global conclusion: to maximize final results, develop both retrieval/context construction and generator/context-to-answer quality.
- Retrieval-only optimization can hit a ceiling if the small generator cannot synthesize the answer.
- Generator-only upgrades can still fail if retrieved evidence is incomplete/noisy.
- The strongest next research direction is a balanced pipeline: semantic chunking + robust dense retriever (MiniLM/E5/BGE) + cross-encoder reranker + better direct/evidence-aware generator.

## Method-Level Conclusions

- `semantic_chunking_dense` had strong answer F1/relevancy early, but weaker grounding metrics; inspect prediction-level behavior before claiming it as best grounded RAG.
- `dense_reranker` is a defensible retrieval improvement because it raises Answer Recall@5, Context Recall, and Context Precision.
- `semantic_chunking_reranker` and MiniLM advanced variants are currently the best practical family for answer quality/runtime tradeoff.
- MiniLM semantic reranker variants beat heavier E5/BGE/GTE variants on F1 in current runs, even when those heavier retrievers improve context metrics.
- `retrieve_k=30/top_k=8` can raise recall but may dilute context and hurt F1.
- Strict/extractive prompts help or preserve F1; citation prompt hurts F1 badly in current setup.
- Neighbor expansion did not help MiniLM/E5 in current runs.
- Qwen direct generation improves answer relevancy compared with Flan-T5-base, but increases runtime.
- Qwen evidence filter/compressor with hard routing produced misleadingly high faithfulness and low relevancy; do not present it as successful.
- RAPTOR is not cost-effective yet for this setup; revisit only if cheaper semantic reranker/generator directions are exhausted.

## Important Implemented Variants

- Independent baselines: `base_dense`, `bm25_only`, `dense_u_shape`, `dense_recency_heavy`, `hybrid_rrf`.
- Advanced retrieval/chunking: `semantic_chunking_dense`, `dense_reranker`, `semantic_chunking_reranker`, `semantic_chunking_hybrid_reranker`.
- RAPTOR family: `raptor_extractive`, `raptor_leiden_abstractive`, `raptor_gmm_abstractive`, `raptor_agglomerative_abstractive`, `semantic_raptor_leiden_reranker`.
- Semantic reranker ablations: MiniLM, E5, BGE, GTE, strict/extractive/citation prompts, neighbor expansion, wide late-chunking, retrieve_k/top_k variants.
- SELF-ROUTE abstention variants: `self_route_minilm_abstain`, `self_route_e5_abstain`.
- Qwen evidence filter variants: `e5_qwen_filter_flan_base`, `e5_qwen_filter_flan_large`, `e5_qwen_soft_route_flan_large`, `e5_qwen_compress_only_flan_large`, `e5_qwen_answer_only`.
- Diagnostic/generator variants: `oracle_gold_context_flan_base`, `oracle_gold_context_flan_base_generator_boost` (professional RAG prompt + U-tail/U-shape ordering + `ANSWER_CRITICAL_EVIDENCE` + `max_input_tokens=4096` + beam4), `sem_rerank_minilm_qwen05_direct`, `sem_rerank_minilm_qwen15_direct`, `contextual_sem_rerank_minilm_flan_base`.

## Verification History To Preserve

- Source modules and notebooks were repeatedly checked with `python -m py_compile`, `python -m unittest discover -s tests`, and notebook code-cell compilation.
- Most recent broad verification: 58 unit tests passed; 348 notebook code cells compiled across 39 notebooks with 0 failures after adding MiniLM/E5 wide late-chunking variants.
- Local environment may not have all ML dependencies such as `sentence_transformers`; full model evaluation is usually performed on Kaggle.

## Answering Guidance

- When the user asks what a variant is, answer using the mapping above and explain its pipeline.
- When the user asks why oracle metrics are high, say it uses gold evidence and bypasses retrieval.
- When comparing methods, separate:
  - answer quality: Token F1 and Relevancy,
  - retrieval quality: Answer Recall@5, Context Recall, Context Precision,
  - grounding: Faithfulness,
  - practicality: Runtime and sec/example.
- For the current narrative, emphasize the joint bottleneck: retrieval must find/pack evidence well, and the generator must convert evidence into the correct answer.
