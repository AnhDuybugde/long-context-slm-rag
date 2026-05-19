# Agent Memory

Last update time: 2026-05-19 Asia/Bangkok

Read this file before doing something new. If a task is verified successfully, update this file with new confirmed information.

## Repository Files

Current local workspace appears to be a fresh or mostly empty project directory. Git metadata was not detected from `git status`, and no application files were confirmed yet because some PowerShell filesystem reads failed in the sandbox with `CreateProcessAsUserW failed: 1312`.

| Path | Main purpose | Expected output | Workflow | Last update |
| --- | --- | --- | --- | --- |
| `.agent/memory.md` | Persistent working memory for file roles, expected outputs, workflow notes, verified external context, and successful task updates. | A concise but growing map of the project and research context. | Read before new work; update after successful verification or meaningful repo changes. | 2026-05-19 |
| `.agent/skills.md` | Local coding principles and behavior rules for this project. | A short set of reminders: think before code, test before code, code simply. | Read before implementation; use as a lightweight checklist during changes. | 2026-05-19 |

## Verified NotebookLM / NLM Context

- `nlm` CLI is installed and responds.
- MCP resource listing returned no exposed resources/templates in this session, but the `nlm` CLI can access NotebookLM data.
- Most recently updated notebook relevant to this task:
  - Title: `RAG`
  - ID: `d92a06a6-76d3-4988-ab1d-0372f9ff45fa`
  - Source count: `56`
  - Updated at: `2026-05-19T09:03:43Z`
- Important notes found in the `RAG` notebook:
  - `Summary Long Context`: long-context windows do not eliminate positional bias; SLMs are especially vulnerable to recency bias; RAG is necessary for cost/performance.
  - `Long-Context Gap`: emphasizes Lost in the Middle, Retrieval Meets Long Context LLMs, LC vs RAG, LOFT, and self-consistency failure under positional bias.
  - `Processing & Retrieval`: covers RAPTOR, LongRAG, advanced retrieval, and datasets/benchmarks.
  - `Environmental Technical Data`: emphasizes semantic chunking, domain-aware/scientific/geospatial embeddings, hybrid search, table parsing, and technical environmental documents.
  - `Tự note`: practical guidance on semantic chunking, Ragas, Vietnamese sentence-transformer embeddings, TF-IDF + dense embeddings, reranking, hierarchical priority, LlamaParse/noisy PDFs, grounding, and model choice.

## Research Direction

The user's current research direction is: implementing a long-context-with-SLM system using RAG. The eventual target can include environmental technical documents or customized data, but the first phase needs a standard, comparable benchmark. The initial dataset should be Qasper from https://allenai.org/data/qasper.

Current priority: complete a base RAG version with a small LLM, then evaluate it. Advanced methods should be added later as incremental improvements so results can prove the direction is correct. Keep clear separation between baseline and future improvements.

## Verified Architecture Direction From Notebook Query

Notebook query on `RAG` verified a recommended pipeline:

`Raw documents -> indexing with semantic chunking + RAPTOR -> SELF-ROUTE -> hybrid retriever -> cross-encoder reranker -> U-shaped context reordering/compression -> SLM generation -> optional long-context fallback`.

Key implementation choices:

- Use semantic chunking for environmental/legal/technical PDFs to preserve formulas, tables, regulation clauses, and scientific argument boundaries.
- Use RAPTOR-style recursive embed-cluster-summarize indexing for multi-level retrieval across details and summaries.
- Use hybrid retrieval: dense vector search for semantic meaning plus BM25/sparse retrieval for exact terms, IDs, standards, measurements, and abbreviations.
- Fuse retrieval lists with RRF or controlled interleaving.
- Rerank retrieved candidates with a cross-encoder such as BGE reranker; use a funnel like retrieve top 50 then rerank/select top 5.
- Avoid overusing MMR for technical corpora where similar-looking passages may contain complementary evidence.
- Reorder context in a U-shape so the most important chunks appear at the beginning and end of the SLM prompt.
- Use SELF-ROUTE: the SLM answers from RAG context when grounded; if insufficient, return `Unanswerable` and route to a stronger long-context model.
- Evaluate with RAGAS: context precision, context recall, faithfulness, answer relevancy, plus cost/latency and failure-point analysis.

## User Corrections On 2026-05-19

- Phase 1 dataset: use Qasper as the standard benchmark. Later phases can try famous datasets or custom data.
- Build a base RAG first and evaluate it before adding advanced improvements.
- Code should be in `.py` files, with a separate folder for `.ipynb` notebooks so training/evaluation can run on Kaggle or Colab.
- The current deliverable must clearly explain what the base RAG includes and what it intentionally does not improve yet.
- Development workflow: code `.py` files first, test them, fix failures until success, then move to the next step. After source code is stable, summarize it into `.ipynb`. Prefer OOP where it helps keep components swappable.

## Implemented Baseline Scaffold On 2026-05-19

Created a base Qasper RAG scaffold:

- `requirements.txt`
- `README.md`
- `src/qasper_base_rag/chunking.py`
- `src/qasper_base_rag/data.py`
- `src/qasper_base_rag/retriever.py`
- `src/qasper_base_rag/generator.py`
- `src/qasper_base_rag/metrics.py`
- `src/qasper_base_rag/pipeline.py`
- `src/qasper_base_rag/evaluate.py`
- `notebooks/qasper_base_rag_colab.ipynb`
- `docs/baseline_scope.md`

Verified with `python -m py_compile` for all source modules. Did not run full dataset/model evaluation locally because it would require downloading Qasper and Hugging Face models.

## GitHub Push On 2026-05-19

- Initialized local git repository.
- Created branch `main`.
- Added remote `origin`: `https://github.com/AnhDuybugde/long-context-slm-rag.git`
- Pushed initial baseline commit `85be98b` to `origin/main`.

## Notebook Update On 2026-05-19

- User could not import the GitHub repo into Kaggle and only needed the Qasper dataset loading step.
- Rewrote `notebooks/qasper_base_rag_colab.ipynb` as a standalone Kaggle/Colab notebook.
- The notebook now uses only `from datasets import load_dataset` and `load_dataset("allenai/qasper")`, inspects splits/schema/sample QA pairs, wraps a sample record in a small OOP helper, and exports a validation preview JSONL.

## Qasper Loading Fix On 2026-05-19

- Kaggle can install a new `datasets` version where dataset scripts are no longer supported, causing `RuntimeError: Dataset scripts are no longer supported, but found qasper.py`.
- Fix: pin `datasets>=2.19.0,<4.0.0` and use `load_dataset("allenai/qasper", trust_remote_code=True)`.
- The HF token warning is not the root error; it only warns about unauthenticated rate limits.
