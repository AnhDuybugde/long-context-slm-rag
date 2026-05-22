# Agent Memory

Last update time: 2026-05-20 Asia/Bangkok

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
- First attempted fix was to pin `datasets>=2.19.0,<4.0.0` and use `trust_remote_code=True`, but Kaggle still reported that `trust_remote_code` is no longer supported.
- Stable fix: load the standard Qasper Parquet exports directly via `load_dataset("parquet", data_files=...)`.
- The HF token warning is not the root error; it only warns about unauthenticated rate limits.

## Qasper Kaggle Output Read On 2026-05-19

- User-provided notebook output confirmed Parquet loading works.
- Loaded splits: train `888`, validation `281`, test `416`.
- Dataset features: `id`, `title`, `abstract`, `full_text`, `qas`, `figures_and_tables`.
- Validation sample `1912.01214` has title `Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine Translation`, 4 QA pairs, and 17 full-text sections.
- Important schema finding: for Parquet records, `qas["answers"][i]` is a dict of columns (`answer`, `annotation_id`, `worker_id`), not a plain list of answer records. `data.py` and notebook helpers must normalize this format before iterating answers.
- Updated Kaggle output confirmed answer normalization works: `show_qa()` now prints full answer records, including `extractive_spans`, `free_form_answer`, `evidence`, `highlighted_evidence`, `annotation_id`, and `worker_id`.
- Next implementation step can move from dataset exploration to the baseline RAG runner/evaluator.

## Base RAG Training Code On 2026-05-19

- Added OOP `BaseRAGTrainer` in `src/qasper_base_rag/trainer.py`.
- Baseline training means building per-document RAG indexes and running SLM generation/evaluation; no model weight fine-tuning yet.
- `evaluate.py` now acts as a CLI wrapper around `BaseRAGTrainer`.
- Trainer writes both JSONL predictions and JSON summary artifacts.
- Added trainer unit test with a fake pipeline so tests do not download models.
- Updated notebook with an optional Base RAG Training Runner section that calls the `.py` CLI after repo code is available in Kaggle/Colab.

## Kaggle Self-Contained Runner On 2026-05-19

- User reported Kaggle cannot import from repo files.
- Updated `notebooks/qasper_base_rag_colab.ipynb` to include a self-contained baseline RAG runner instead of requiring `src/qasper_base_rag` imports.
- The notebook now defines local OOP/data classes for chunks, QA examples, dense retriever, small generator, and base pipeline.
- The `.py` files remain the source of truth and are tested; the notebook is a Kaggle-friendly runnable copy.

## Full Baseline Run Policy On 2026-05-19

- User clarified that real baseline training/evaluation should run the full split, not stop after a small debug limit.
- Updated CLI/trainer/notebook so `limit=None` runs the full selected split.
- `--limit` / `limit=5` should be used only for smoke tests.

## Base RAG Full Validation Result On 2026-05-19

User ran the self-contained Kaggle baseline on full Qasper validation.

- Examples: `1005`
- `avg_token_f1`: `0.13836707101945536`
- `avg_answer_string_recall_at_5`: `0.22354892205638474`
- `avg_context_precision`: `0.7122388059701488`
- `avg_context_recall`: `0.8378653914553359`
- `avg_faithfulness`: `0.9862354892205638`
- `avg_answer_relevancy`: `0.09799809183209088`
- Predictions path on Kaggle: `outputs/base_rag_qasper_predictions.jsonl`

Interpretation: baseline dense retrieval retrieves useful context reasonably well, but the small generator/prompt produces weak answer overlap and low answer relevancy. This supports continuing with long-context/RAG improvements, especially reranking, context packing/reordering, better generation prompting, and later advanced indexing.

## First Improved RAG Variant On 2026-05-19

- Added `src/qasper_base_rag/improved.py` with BM25 sparse retrieval, RRF fusion, U-shaped context reordering, and stricter evidence-focused generation prompt.
- Added tests for tokenization, BM25 keyword matching, RRF, and U-shaped reordering.
- Added `notebooks/qasper_improved_rag_colab.ipynb`, a self-contained Kaggle notebook for the improved variant.
- This variant should be compared to the baseline full validation metrics before adding heavier methods like cross-encoder reranking or RAPTOR.

## Phase 2 Output Confirmed On 2026-05-20

User reported phase 2 is complete. Local `notebooks/output/` contains five standalone output notebooks:

- `environmental-rag-template-standalone-colab.ipynb`
- `lcslm-improved-rag.ipynb`
- `qasper-ablation-standalone-colab.ipynb`
- `qasper-research-rag-standalone-colab.ipynb`
- `qasper-self-route-standalone-colab.ipynb`

Treat these as the current phase 2 artifacts before planning phase 3. The MCP resource listing still returned no exposed NotebookLM resources in this session, but `nlm --help` works and the CLI remains the practical NotebookLM access path if notebook context is needed.

## Research Priority Correction On 2026-05-20

Current priority is not to build one hybrid/best-of-all pipeline yet. The user wants to run methods independently on the long-context/SLM RAG setting, compare their runtime and metrics, identify which method is best, and only then go deeper into that selected method for further development.

Implication for phase 3 planning:

- Keep experimental variants isolated and comparable.
- Avoid prematurely combining all strong-looking techniques into a single pipeline.
- Treat baseline, context reordering, RAPTOR, SELF-ROUTE, semantic chunking, reranking, long-context fallback, and environmental/table/geospatial hooks as separable experiments unless the user explicitly asks for integration.
- Prioritize clean ablation tables, fair configs, runtime/metric comparison, and failure analysis.

## Long-Context Data Survey On 2026-05-20

Added source CLIs for independent method runs and long-context data checking:

- `src/qasper_base_rag/survey_long_context.py`: surveys Qasper paper word counts and threshold coverage.
- `src/qasper_base_rag/evaluate_experiment.py`: runs one independent variant at a time with optional `--min-doc-words` filtering.
- `src/qasper_base_rag/experiment_pipelines.py`: separate method pipelines for `base_dense`, `bm25_only`, `dense_u_shape`, `dense_recency_heavy`, and `hybrid_rrf`.

Validation split survey output:

- Documents: `281`
- QA examples: `1005`
- Word counts: min `650`, p25 `2395`, median `3473`, mean `3638.41`, p75 `4337`, p90 `5030`, p95 `6393`, max `14882`
- Threshold coverage:
  - `>=1000` words: `280` docs (`99.6%`), `1003` QA (`99.8%`)
  - `>=3000` words: `165` docs (`58.7%`), `583` QA (`58.0%`)
  - `>=5000` words: `29` docs (`10.3%`), `94` QA (`9.4%`)
  - `>=8000` words: `9` docs (`3.2%`), `30` QA (`3.0%`)
  - `>=12000` words: `4` docs (`1.4%`), `14` QA (`1.4%`)

Interpretation: Qasper is acceptable for long-document scientific-paper RAG, especially with a `--min-doc-words 3000` filter. It is not a pure ultra-long-context benchmark; very long papers above 8k-12k words are a small subset. For phase 3, report both full validation and filtered long-context results.

## Kaggle Independent Notebooks On 2026-05-20

User trains on Kaggle and cannot import local repo files there. Created self-contained notebooks under `notebooks/independent_variants/`; each includes all code it needs and runs one method independently:

- `qasper_long_context_survey_standalone.ipynb`
- `qasper_base_dense_standalone.ipynb`
- `qasper_bm25_only_standalone.ipynb`
- `qasper_dense_u_shape_standalone.ipynb`
- `qasper_dense_recency_heavy_standalone.ipynb`
- `qasper_hybrid_rrf_standalone.ipynb`

Defaults in experiment notebooks: `SPLIT="validation"`, `MIN_DOC_WORDS=3000`, `LIMIT=None`, outputs under `outputs/independent/`. Change `LIMIT=10` for a smoke test. Verified the notebooks parse/compile and do not import from `src` or `qasper_base_rag`.

## Independent Variant Coverage Audit On 2026-05-20

Reviewed `.agent/skills.md`, `docs/independent_experiments.md`, `src/qasper_base_rag/experiment_pipelines.py`, `src/qasper_base_rag/chunking.py`, `src/qasper_base_rag/metrics.py`, `src/qasper_base_rag/trainer.py`, and `notebooks/independent_variants/`.

Current independent variants cover only the first ablation layer:

- Fixed word chunking baseline, not semantic chunking.
- Dense retrieval baseline.
- BM25-only sparse retrieval.
- Dense + U-shaped context reordering.
- Dense + recency-heavy ordering for SLM positional-bias comparison.
- Dense + BM25 RRF as a comparison method.
- Long-document filtering/survey via Qasper word-count thresholds.

Current independent variants do not yet cover RAPTOR, semantic chunking, cross-encoder reranking, SELF-ROUTE, real long-context fallback, Leiden/adaptive graph clustering, GeoFM/domain embeddings, table/SQL parsing, or environmental-specific data. `notebooks/output/` contains older/prototype notebooks for SELF-ROUTE/research-RAG, but these are not part of the current isolated `notebooks/independent_variants/` suite.

Verified with `python -m unittest discover -s tests` on 2026-05-20: 18 tests passed.

## Advanced Independent Notebooks Added On 2026-05-20

User asked to add the missing methods as standalone notebooks so they can train/run them in parallel on Kaggle.

Added source implementation in `src/qasper_base_rag/advanced_variants.py` and wired these variants into `src/qasper_base_rag/evaluate_experiment.py` / `src/qasper_base_rag/experiment_pipelines.py`:

- `semantic_chunking_dense`: semantic sentence-boundary chunking plus dense retrieval.
- `dense_reranker`: dense retrieval followed by cross-encoder reranking, with lexical fallback if the reranker cannot load.
- `raptor_extractive`: RAPTOR-style collapsed tree with extractive parent summaries and graph-similarity grouping; this is a proxy, not full LLM-abstractive RAPTOR/Leiden.
- `raptor_leiden_abstractive`: more faithful RAPTOR with recursive abstractive summaries and Leiden graph clustering when `igraph`/`leidenalg` install successfully.

Added `scripts/build_independent_variant_notebooks.py` to regenerate the advanced standalone notebooks from the base independent notebook template.

New notebooks in `notebooks/independent_variants/`:

- `qasper_semantic_chunking_dense_standalone.ipynb`
- `qasper_dense_reranker_standalone.ipynb`
- `qasper_raptor_extractive_standalone.ipynb`
- `qasper_raptor_leiden_abstractive_standalone.ipynb`

Verification on 2026-05-20:

- `python -m unittest discover -s tests`: 25 tests passed.
- All code cells in `notebooks/independent_variants/qasper_*_standalone.ipynb` compile after filtering notebook shell-magics.
- All standalone notebooks were checked to avoid imports from `src` / `qasper_base_rag`.

## Scope Reduction On 2026-05-20

User decided to exclude SELF-ROUTE, GeoFM, and Table/SQL from the independent comparison because adding Gemini/API fallback would be unfair and resource-heavy.

Removed from the active independent experiment suite:

- `self_route_extractive`
- `structured_table_router`
- `qasper_self_route_extractive_standalone.ipynb`
- `qasper_structured_table_router_standalone.ipynb`

Also removed the SELF-ROUTE and structured/table-router classes from `src/qasper_base_rag/advanced_variants.py` and the notebook generator so the remaining generated notebooks no longer carry unused SELF-ROUTE/table/geospatial code.

Current active independent variants after reduction:

- `base_dense`
- `bm25_only`
- `dense_u_shape`
- `dense_recency_heavy`
- `hybrid_rrf`
- `semantic_chunking_dense`
- `dense_reranker`
- `raptor_extractive`
- `raptor_leiden_abstractive`

Verification after reduction:

- `python -m unittest discover -s tests`: 22 tests passed.
- All active `notebooks/independent_variants/qasper_*_standalone.ipynb` code cells compile after filtering notebook shell-magics.
- Active source/notebook generator no longer contains SELF-ROUTE, structured-table, or geospatial router code except historical notes in docs/memory.
- Regenerated active advanced notebooks again to remove stale `fallback_rate` / `long_context_fallback` summary fields after SELF-ROUTE removal.

## Active Notebook Run Plan On 2026-05-20

Active Kaggle standalone notebooks for the independent comparison:

| Notebook | Variant | Purpose | Main output |
| --- | --- | --- | --- |
| `qasper_long_context_survey_standalone.ipynb` | survey | Check Qasper document-length coverage. | Word-count thresholds and long-context subset sizes. |
| `qasper_base_dense_standalone.ipynb` | `base_dense` | Dense retrieval baseline. | Baseline metrics and predictions. |
| `qasper_bm25_only_standalone.ipynb` | `bm25_only` | Sparse keyword retrieval baseline. | BM25-only metrics and predictions. |
| `qasper_dense_u_shape_standalone.ipynb` | `dense_u_shape` | Test U-shaped context reordering for lost-in-the-middle mitigation. | Reordered-context metrics and predictions. |
| `qasper_dense_recency_heavy_standalone.ipynb` | `dense_recency_heavy` | Test SLM recency-bias-friendly ordering. | Recency-order metrics and predictions. |
| `qasper_hybrid_rrf_standalone.ipynb` | `hybrid_rrf` | Dense + BM25 with reciprocal rank fusion. | Hybrid retrieval metrics and predictions. |
| `qasper_semantic_chunking_dense_standalone.ipynb` | `semantic_chunking_dense` | Test semantic sentence-boundary chunking. | Semantic-chunk metrics and predictions. |
| `qasper_dense_reranker_standalone.ipynb` | `dense_reranker` | Dense candidate retrieval plus cross-encoder reranking. | Reranker metrics, predictions, and reranker load metadata. |
| `qasper_raptor_extractive_standalone.ipynb` | `raptor_extractive` | RAPTOR-style hierarchy with extractive summaries. | RAPTOR-proxy metrics and predictions. |
| `qasper_raptor_leiden_abstractive_standalone.ipynb` | `raptor_leiden_abstractive` | More faithful RAPTOR with abstractive parent summaries and Leiden if available. | RAPTOR-Leiden metrics, predictions, parent count, and backend metadata. |

All experiment notebooks default to `SPLIT="validation"`, `MIN_DOC_WORDS=3000`, `LIMIT=None`, and write to `outputs/independent/`. Use `LIMIT=10` first for smoke tests, then run full. After running all notebooks, collect each `*_summary.json` into one comparison table with F1, answer string recall, context precision/recall, faithfulness, answer relevancy, runtime, and seconds/example. Then inspect `*_predictions.jsonl` for the winning and losing variants, choose the best method, and tune only that method next.

## Kaggle/Colab NumPy Import Fix On 2026-05-20

User reported `ImportError: cannot import name '_center' from 'numpy._core.umath'`, matching a known `sentence_transformers -> sklearn -> scipy -> numpy` import failure in Python 3.12 notebooks when loose dependency pins pull incompatible/broken scientific stack versions.

Fixed all active `notebooks/independent_variants/` setup cells by pinning:

- `numpy==1.26.4`
- `scipy==1.13.1`
- `scikit-learn==1.5.2`
- `sentence-transformers==3.0.1`
- `transformers==4.44.2`
- `datasets>=2.19.0,<4.0.0`

Also updated `requirements.txt` and `scripts/build_independent_variant_notebooks.py` so regenerated notebooks keep the same pins. The setup cell now notes that if the error already happened in the current notebook session, restart the Kaggle/Colab runtime before rerunning from the top.

Verification after fix:

- All active notebook code cells compile after filtering shell-magics.
- `python -m unittest discover -s tests`: 22 tests passed.

Follow-up fix in the `.ipynb` files themselves:

- Added a dependency sanity check directly after the pip install lines in every active `notebooks/independent_variants/*.ipynb`.
- The check imports `numpy`, `scipy`, `sklearn`, and `SentenceTransformer`, verifies `numpy==1.26.4`, and prints package versions.
- If Kaggle/Colab still has an old NumPy module loaded, the notebook now fails early with a clear instruction to restart the session and rerun from the first cell.

## Kaggle Kernel Die Fix On 2026-05-20

User reported that running the standalone independent variant notebooks killed the kernel and training could not start.

Root cause found locally: setup cells in `notebooks/independent_variants/*.ipynb` still called `os._exit(0)` after package install/uninstall and still pinned `numpy==2.0.2`, contradicting the stable NumPy fix recorded above. This made Kaggle/Colab look like the kernel died, and could also reintroduce binary-stack instability.

Fix applied:

- Removed `os._exit(0)` from all active standalone notebooks; setup cells now raise `SystemExit` with a manual restart instruction instead of force-killing the process.
- Restored stable dependency pins in notebooks and `requirements.txt`: `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.5.2`.
- Updated `scripts/build_independent_variant_notebooks.py` so regenerated advanced notebooks keep the same safe setup behavior and dependency pins.
- Added `model.eval()` and `torch.inference_mode()` to `SmallSeq2SeqGenerator`; generation now uses `num_beams=1` to lower VRAM/RAM pressure during full validation.
- Added `scripts/repair_independent_notebooks.py` as a structured JSON notebook repair helper for applying these runtime-safety fixes across all independent notebooks.

Verification:

- `Select-String` found no remaining `os._exit`, `numpy==2.0.2`, or `num_beams=2` in active notebooks/generator script.
- All active independent notebook code cells compile.
- `python -m unittest discover -s tests`: 22 tests passed.
- `python -m py_compile src/qasper_base_rag/generator.py scripts/build_independent_variant_notebooks.py scripts/repair_independent_notebooks.py`: passed.

## Kaggle Simple Setup Correction On 2026-05-20

User reported Kaggle dependency resolver conflicts after the pinned setup downgraded NumPy/scikit-learn and removed `torchvision`. Current correction:

- Active independent notebooks now use a simple setup cell: no forced reinstall, no NumPy/SciPy/scikit-learn pins, no `torchvision` uninstall, no `SystemExit`, no restart requirement.
- The setup cell only checks required imports and runs `pip install -q <missing packages>` if a package is absent.
- `raptor_leiden_abstractive` still includes optional `igraph`/`leidenalg` in the missing-package check; other notebooks avoid those extras.
- `requirements.txt` now uses broad scientific stack ranges instead of exact pins: `numpy>=1.26,<3`, `scipy>=1.13,<2`, `scikit-learn>=1.5,<2`.
- Verification: no active notebook or notebook builder contains `force-reinstall`, `SystemExit`, `os._exit`, `numpy==`, `scipy==`, `scikit-learn==`, `torchvision`, or `--no-deps`; all active notebook code cells compile.

## Survey Notebook JSONL Output On 2026-05-20

User reported `qasper_long_context_survey_standalone` only produced the aggregate JSON, not JSONL.

Updated `notebooks/independent_variants/qasper_long_context_survey_standalone.ipynb` so `run_long_context_survey()` writes:

- `outputs/independent/qasper_long_context_survey_validation.json`: aggregate summary.
- `outputs/independent/qasper_long_context_survey_validation_documents.jsonl`: one row per Qasper document with `split`, `doc_id`, `title`, `word_count`, `qa_examples`, and threshold boolean flags.
- `outputs/independent/qasper_long_context_survey_validation_thresholds.jsonl`: one row per threshold with document/QA counts and rates.

The returned summary now includes `summary_path`, `documents_jsonl_path`, and `thresholds_jsonl_path`. Verified the survey notebook code cells compile.

## Independent Variant Results On 2026-05-20

User ran all active independent Qasper validation notebooks with `MIN_DOC_WORDS=3000`, `LIMIT=None`. Local `notebooks/output/` contains the nine `*_validation_min3000_summary.json` files for active variants.

Shared run context:

- Split: `validation`
- Minimum document length: `3000` words
- Documents seen: `165`
- QA examples: `583`

Comparison table from `notebooks/output/*validation_min3000_summary.json`:

| Variant | F1 | Answer Recall@5 | Context Recall | Context Precision | Faithfulness | Relevancy | Runtime | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base_dense` | 0.1165 | 0.2018 | 0.4147 | 0.6021 | 0.5472 | 0.0411 | 179.79s | Baseline is stable. |
| `bm25_only` | 0.0891 | 0.1795 | 0.3504 | 0.6153 | 0.5437 | 0.0293 | 145.84s | Fast, but weaker F1 and recall. |
| `dense_u_shape` | 0.1043 | 0.2018 | 0.4147 | 0.6021 | 0.5232 | 0.0407 | 177.60s | U-shaped ordering did not improve over baseline. |
| `dense_recency_heavy` | 0.1047 | 0.2018 | 0.4147 | 0.6021 | 0.5146 | 0.0426 | 169.66s | Recency-heavy ordering did not clearly help. |
| `hybrid_rrf` | 0.1021 | 0.2053 | 0.4280 | 0.6343 | 0.5643 | 0.0397 | 186.96s | Slight retrieval/faithfulness gain, but no F1 gain. |
| `semantic_chunking_dense` | 0.2256 | 0.1561 | 0.3279 | 0.5300 | 0.3722 | 0.1356 | 221.20s | Best answer F1/relevancy, but low retrieval and faithfulness metrics need prediction-level inspection. |
| `dense_reranker` | 0.1190 | 0.2207 | 0.4603 | 0.6518 | 0.5506 | 0.0456 | 210.81s | Best retrieval quality; strong candidate for further RAG tuning. |
| `raptor_extractive` | 0.1425 | 0.1838 | 0.3719 | 0.5743 | 0.4803 | 0.0755 | 3043.50s | F1 improves, but runtime is very expensive. |
| `raptor_leiden_abstractive` | 0.1522 | 0.1787 | 0.3521 | 0.4868 | 0.4648 | 0.0864 | 1655.36s | Better F1 than baseline, but slower and weaker retrieval. |

Current interpretation:

- If optimizing directly for answer F1, `semantic_chunking_dense` is the top result, but its low context recall/precision and faithfulness mean this result should be checked in `*_predictions.jsonl` before choosing it as the main research direction.
- If choosing the most defensible RAG improvement based on retrieval quality, `dense_reranker` is the strongest next method: best Answer Recall@5, Context Recall, and Context Precision.
- `hybrid_rrf` is useful evidence that sparse+dense fusion helps retrieval slightly, but the current generator does not convert that into answer F1.
- RAPTOR variants are not cost-effective yet for this setup; they may be revisited only after cheaper methods are exhausted.
- `dense_u_shape` and `dense_recency_heavy` have no clear positive signal in this batch.

Recommended next step: inspect prediction-level failure cases for `semantic_chunking_dense` and `dense_reranker`, then choose one method to tune further. For a conservative research narrative, `dense_reranker` is currently the best method to develop next; for a metric-winning narrative, investigate whether `semantic_chunking_dense` is genuinely answering better or exploiting metric artifacts.

## Output Summary Report On 2026-05-21

Created `notebooks/output/independent_validation_min3000_summary.md` as the consolidated report for the local Kaggle outputs.

Additional prediction-level checks from available `*_predictions.jsonl` files:

- `semantic_chunking_dense`: lowest `Unanswerable` rate among checked variants (`132/583`, 0.23) and highest rate of examples with Token F1 > 0 (`288/583`, 0.49). This explains its strong F1, but the low context recall/precision/faithfulness still requires qualitative inspection before treating it as the best grounded RAG method.
- `dense_reranker`: strongest retrieval-oriented method; highest available count of examples with Answer Recall@5 > 0 (`178/583`) and best aggregate Answer Recall@5, Context Recall, and Context Precision.
- Local output contains `raptor_extractive_validation_min3000_summary.json`, but no `raptor_extractive_validation_min3000_predictions.jsonl`, so prediction-level stats for `raptor_extractive` were not included in the report.

Current reporting recommendation: present a two-track conclusion. `semantic_chunking_dense` is the metric winner for lexical answer quality, while `dense_reranker` is the more defensible next method for grounded RAG tuning. RAPTOR variants are currently too slow relative to their gains.

## Phase 3 Combined Variant Added On 2026-05-22

Added a follow-up combined method after the independent comparison:

- Variant: `semantic_chunking_reranker`
- Standalone notebook: `notebooks/independent_variants/qasper_semantic_chunking_reranker_standalone.ipynb`
- Pipeline: semantic sentence-boundary chunking -> dense retrieval with `retrieve_k=20` -> cross-encoder reranking -> top 5 contexts -> `google/flan-t5-base` generation.
- Purpose: test whether the high answer F1/relevancy signal from `semantic_chunking_dense` can be combined with the stronger retrieval/context precision signal from `dense_reranker`.

Source wiring added in `src/qasper_base_rag/advanced_variants.py`, `src/qasper_base_rag/experiment_pipelines.py`, `src/qasper_base_rag/evaluate_experiment.py`, and `scripts/build_independent_variant_notebooks.py`.

Verification:

- `python -m py_compile src/qasper_base_rag/advanced_variants.py src/qasper_base_rag/experiment_pipelines.py src/qasper_base_rag/evaluate_experiment.py scripts/build_independent_variant_notebooks.py`: passed.
- `python -m unittest discover -s tests`: 23 tests passed.
- New standalone notebook code cells compile after filtering notebook shell-magics.

## Phase 3 Hybrid Reranker Variant Added On 2026-05-22

Added the next follow-up combined method requested by the user:

- Variant: `semantic_chunking_hybrid_reranker`
- Standalone notebook: `notebooks/independent_variants/qasper_semantic_chunking_hybrid_reranker_standalone.ipynb`
- Pipeline: semantic sentence-boundary chunking -> dense retrieval + BM25 retrieval -> Reciprocal Rank Fusion candidates -> cross-encoder reranking -> top-k contexts -> `google/flan-t5-base` generation.
- Tuning knobs exposed in the notebook/config/CLI: `RETRIEVE_K` / `--retrieve-k` and `TOP_K` / `--top-k`.
- Purpose: test whether the semantic chunking answer-quality signal can be combined with hybrid RRF recall and reranker precision.

Source wiring added in `src/qasper_base_rag/advanced_variants.py`, `src/qasper_base_rag/experiment_pipelines.py`, `src/qasper_base_rag/evaluate_experiment.py`, `scripts/build_independent_variant_notebooks.py`, and `notebooks/README.md`.

Verification:

- `python -m py_compile src/qasper_base_rag/advanced_variants.py src/qasper_base_rag/experiment_pipelines.py src/qasper_base_rag/evaluate_experiment.py scripts/build_independent_variant_notebooks.py`: passed.
- `python -m unittest discover -s tests`: 24 tests passed.
- New standalone notebook code cells compile after filtering notebook shell-magics.

## RAPTOR Development Variants Added On 2026-05-22

Added three RAPTOR follow-up variants based on the user's requested research directions:

- `raptor_gmm_abstractive`: original RAPTOR-style fixed leaf chunks -> GMM clustering -> abstractive parent summaries -> collapsed-tree dense retrieval.
- `raptor_agglomerative_abstractive`: fixed leaf chunks -> position-aware agglomerative clustering -> abstractive parent summaries -> deeper collapsed-tree dense retrieval.
- `semantic_raptor_leiden_reranker`: semantic leaf chunking -> adaptive-threshold Leiden RAPTOR parent summaries -> collapsed-tree dense retrieval -> cross-encoder reranking.

New standalone notebooks:

- `notebooks/independent_variants/qasper_raptor_gmm_abstractive_standalone.ipynb`
- `notebooks/independent_variants/qasper_raptor_agglomerative_abstractive_standalone.ipynb`
- `notebooks/independent_variants/qasper_semantic_raptor_leiden_reranker_standalone.ipynb`

Source wiring added in `src/qasper_base_rag/advanced_variants.py`, `src/qasper_base_rag/experiment_pipelines.py`, `src/qasper_base_rag/evaluate_experiment.py`, `scripts/build_independent_variant_notebooks.py`, `notebooks/README.md`, and `tests/test_advanced_variants.py`.

Verification:

- `python -m py_compile src/qasper_base_rag/advanced_variants.py src/qasper_base_rag/experiment_pipelines.py src/qasper_base_rag/evaluate_experiment.py scripts/build_independent_variant_notebooks.py`: passed.
- `python -m unittest discover -s tests`: 26 tests passed.
- The three new standalone notebook code cells compile after filtering notebook shell-magics.

## RAPTOR Paper-Closer Update On 2026-05-22

After web verification against the official RAPTOR GitHub implementation, the Stanford CS224N Laitenberger report, and the Frontiers semantic-chunking/AGC paper, the RAPTOR variants were revised to be closer to the described methods:

- `raptor_gmm_abstractive`: now uses RAPTOR-style UMAP dimensionality reduction when available, BIC-based GMM cluster count selection, soft GMM memberships with threshold `0.10`, global/local clustering, and recursive reclustering for oversized clusters.
- `raptor_agglomerative_abstractive`: now uses average-linkage agglomerative clustering over cosine distances, appends absolute positional features (start position, distance to end, rough section) scaled to the SBERT-like range and priority-weighted, cuts the hierarchy at approximately `n/3` and `n/6`, and adds a top root summary node.
- `semantic_raptor_leiden_reranker`: now uses semantic segmentation closer to tau `0.7` via distance threshold `0.30`, adaptive Leiden with kNN graph construction, layer-aware increasing neighbors, layer-aware decreasing resolution, and reranking over collapsed-tree candidates.
- Evaluation timing now separates indexing/tree-build time from per-question answer/inference time. Source CLI summaries include `total_index_seconds`, `avg_index_seconds_per_doc`, `total_answer_seconds`, and `avg_answer_seconds_per_example`; standalone notebooks include `index_seconds_total`, `index_seconds_per_doc`, `answer_seconds_total`, and `answer_seconds_per_example`, plus per-row `answer_seconds`.

Verification:

- `python -m py_compile src/qasper_base_rag/advanced_variants.py src/qasper_base_rag/trainer.py src/qasper_base_rag/experiment_pipelines.py src/qasper_base_rag/evaluate_experiment.py scripts/build_independent_variant_notebooks.py`: passed.
- `python -m unittest discover -s tests`: 26 tests passed.
- The three RAPTOR standalone notebooks compile after regeneration. Runtime smoke initialization could not run locally because `sentence_transformers` is not installed in the local Python environment.
