"""Create a standalone adaptive chunk/flexible context ablation notebook."""

from __future__ import annotations

import json
from pathlib import Path

from create_cache_memory_notebook import SETUP, code, markdown
from create_evidence_selected_notebook import CORE


NOTEBOOK_PATH = Path("notebooks/18-adaptive-chunk-flexible-context-ablation.ipynb")


ADAPTIVE_CORE = r'''
SPLIT = "validation"
LIMIT = 50
RANDOM_SEED = 13
BASE_CONDITION = "dense_pool20_cross_top8_router"
DENSE_MODEL_NAME = "intfloat/e5-small-v2"
GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


ADAPTIVE_CONDITIONS = [
    "dense_pool20_cross_top8_router",
    "small80_pool20_cross_expand160_top8_router",
    "small80_pool20_cross_expand240_top8_router",
    "small80_pool30_cross_expand160_top8_router",
    "small80_cross_before_expand160_top8_router",
    "small80_expand160_cross_after_top8_router",
    "small80_cross_expand160_cross_top8_router",
    "sentence_pool30_cross_window_pm1_top8_router",
    "sentence_pool30_cross_window_pm2_top8_router",
    "sentence_pool50_cross_window_paragraph_top8_router",
    "paragraph_pool20_cross_top8_router",
    "paragraph_pool30_cross_top8_router",
    "small80_pool30_cross_expand160_budget600_router",
    "small80_pool30_cross_expand160_budget800_router",
    "sentence_pool50_cross_window_pm2_budget700_router",
]


def question_type_from_text(question: str) -> str:
    normalized = normalize_answer(question)
    tokens = normalized.split()
    if not tokens:
        return "free_form"
    first = tokens[0]
    if first in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "has", "have", "had"}:
        return "yes_no"
    if first in {"what", "which", "who", "where", "when", "how"}:
        return "extractive"
    return "free_form"


def context_block(contexts: list[RetrievedContext], *, include_scores: bool = True) -> str:
    if not contexts:
        return "[No context provided]"
    lines = []
    for context in contexts:
        score_part = f" score={context.score:.4f}" if include_scores else ""
        lines.append(f"[{context.rank}] section={context.chunk.section}{score_part}\n{context.chunk.text}")
    return "\n\n".join(lines)


def build_question_type_router_prompt(question: str, contexts: list[RetrievedContext]) -> str:
    question_type = question_type_from_text(question)
    if question_type == "yes_no":
        instruction = "Return only Yes, No, or Unanswerable."
    elif question_type == "extractive":
        instruction = "Return only the shortest evidence-supported answer phrase."
    else:
        instruction = "Return one short evidence-supported answer sentence."
    return (
        "Answer using only the provided retrieved context. "
        f"Question type: {question_type}. {instruction}\n\n"
        f"Context:\n{context_block(contexts)}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def trim_first_line(text: str) -> str:
    text = str(text or "").strip()
    return text.splitlines()[0].strip() if text else ""


def source_items_for_paper(paper: Paper) -> list[dict[str, Any]]:
    items = []
    if paper.title:
        items.append({"label": "title", "text": paper.title, "kind": "title", "index": 0})
    if paper.abstract:
        items.append({"label": "abstract", "text": paper.abstract, "kind": "abstract", "index": 0})
    for section in paper.sections:
        for paragraph_index, paragraph in enumerate(section.paragraphs):
            label = f"{section.name} / paragraph {paragraph_index + 1}"
            items.append({"label": label, "text": paragraph, "kind": "paragraph", "index": paragraph_index})
    return items


def item_text_by_label(paper: Paper) -> dict[str, str]:
    return {item["label"]: item["text"] for item in source_items_for_paper(paper)}


def split_adaptive_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    sentences = []
    for piece in pieces:
        piece = piece.strip(" \t\r\n-")
        words = piece.split()
        if len(words) < 4:
            continue
        if len(words) > 90:
            for start in range(0, len(words), 70):
                span = " ".join(words[start:start + 90]).strip()
                if len(span.split()) >= 4:
                    sentences.append(span)
        else:
            sentences.append(piece)
    return sentences


def build_paragraph_chunks(paper: Paper) -> list[Chunk]:
    chunks = []
    for item in source_items_for_paper(paper):
        text = item["text"].strip()
        if not text:
            continue
        chunks.append(Chunk(
            paper_id=paper.paper_id,
            chunk_id=f"{paper.paper_id}:paragraph:{len(chunks)}",
            section=item["label"],
            text=text,
            title=paper.title,
            start_token=0,
            end_token=len(text.split()),
        ))
    return chunks


def build_sentence_chunks(paper: Paper) -> tuple[list[Chunk], dict[str, dict[str, Any]]]:
    chunks = []
    metadata = {}
    for item_index, item in enumerate(source_items_for_paper(paper)):
        token_cursor = 0
        for sentence_index, sentence in enumerate(split_adaptive_sentences(item["text"])):
            words = sentence.split()
            chunk_id = f"{paper.paper_id}:sentence:{item_index}:{sentence_index}"
            chunk = Chunk(
                paper_id=paper.paper_id,
                chunk_id=chunk_id,
                section=item["label"],
                text=sentence,
                title=paper.title,
                start_token=token_cursor,
                end_token=token_cursor + len(words),
            )
            chunks.append(chunk)
            metadata[chunk_id] = {
                "item_index": item_index,
                "sentence_index": sentence_index,
                "label": item["label"],
                "sentences": split_adaptive_sentences(item["text"]),
                "full_text": item["text"],
            }
            token_cursor += len(words)
    return chunks, metadata


def with_new_ranks(contexts: list[RetrievedContext], source: str | None = None) -> list[RetrievedContext]:
    return [
        RetrievedContext(context.chunk, float(context.score), rank, source or context.source)
        for rank, context in enumerate(contexts, start=1)
    ]


def dedupe_contexts_by_text(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen = set()
    deduped = []
    for context in contexts:
        key = normalize_answer(context.chunk.text)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
    return deduped


def expand_context_to_token_window(example: QAExample, context: RetrievedContext, target_tokens: int, source: str) -> RetrievedContext:
    text_by_label = item_text_by_label(example.paper)
    full_text = text_by_label.get(context.chunk.section, context.chunk.text)
    tokens = full_text.split()
    if not tokens:
        return context
    center = (int(context.chunk.start_token) + int(context.chunk.end_token)) // 2
    half = max(1, target_tokens // 2)
    start = max(0, center - half)
    end = min(len(tokens), start + target_tokens)
    start = max(0, end - target_tokens)
    expanded_text = " ".join(tokens[start:end]).strip()
    chunk = Chunk(
        paper_id=context.chunk.paper_id,
        chunk_id=f"{context.chunk.chunk_id}:{source}:{target_tokens}:{start}:{end}",
        section=context.chunk.section,
        text=expanded_text,
        title=context.chunk.title,
        start_token=start,
        end_token=end,
    )
    return RetrievedContext(chunk, context.score, context.rank, source)


def expand_sentence_window(example: QAExample, context: RetrievedContext, sentence_metadata: dict[str, dict[str, Any]], radius: int, source: str) -> RetrievedContext:
    info = sentence_metadata.get(context.chunk.chunk_id)
    if not info:
        return expand_context_to_token_window(example, context, 160, source)
    sentences = info["sentences"]
    sentence_index = info["sentence_index"]
    start = max(0, sentence_index - radius)
    end = min(len(sentences), sentence_index + radius + 1)
    text = " ".join(sentences[start:end]).strip()
    chunk = Chunk(
        paper_id=context.chunk.paper_id,
        chunk_id=f"{context.chunk.chunk_id}:{source}:pm{radius}",
        section=context.chunk.section,
        text=text,
        title=context.chunk.title,
        start_token=0,
        end_token=len(text.split()),
    )
    return RetrievedContext(chunk, context.score, context.rank, source)


def expand_sentence_to_paragraph(example: QAExample, context: RetrievedContext, source: str) -> RetrievedContext:
    text = item_text_by_label(example.paper).get(context.chunk.section, context.chunk.text)
    chunk = Chunk(
        paper_id=context.chunk.paper_id,
        chunk_id=f"{context.chunk.chunk_id}:{source}:paragraph",
        section=context.chunk.section,
        text=text,
        title=context.chunk.title,
        start_token=0,
        end_token=len(text.split()),
    )
    return RetrievedContext(chunk, context.score, context.rank, source)


def cross_rerank_top(question: str, contexts: list[RetrievedContext], final_top_k: int, source: str) -> list[RetrievedContext]:
    if not contexts:
        return []
    reranked = rerank_with_cross_encoder(question, contexts, CROSS_ENCODER_MODEL_NAME)
    return with_new_ranks(reranked[:final_top_k], source)


def pack_context_budget(contexts: list[RetrievedContext], budget_words: int, source: str) -> list[RetrievedContext]:
    packed = []
    used_words = 0
    for context in contexts:
        words = context.chunk.text.split()
        remaining = budget_words - used_words
        if remaining <= 0:
            break
        if len(words) > remaining:
            if not packed and remaining >= 60:
                text = " ".join(words[:remaining])
                chunk = Chunk(
                    paper_id=context.chunk.paper_id,
                    chunk_id=f"{context.chunk.chunk_id}:{source}:trimmed",
                    section=context.chunk.section,
                    text=text,
                    title=context.chunk.title,
                    start_token=context.chunk.start_token,
                    end_token=context.chunk.start_token + len(text.split()),
                )
                packed.append(RetrievedContext(chunk, context.score, context.rank, source))
                used_words += len(text.split())
            continue
        packed.append(RetrievedContext(context.chunk, context.score, context.rank, source))
        used_words += len(words)
    return with_new_ranks(packed, source)


def context_word_count(contexts: list[RetrievedContext]) -> int:
    return sum(len(context.chunk.text.split()) for context in contexts)


def adaptive_trace(
    *,
    condition: str,
    question: str,
    contexts: list[RetrievedContext],
    prompt: str,
    retrieval_unit: str,
    expansion_policy: str,
    rerank_stage: str,
    packing_budget_words: int,
    candidate_pool_k: int,
    final_top_k: int,
) -> dict[str, Any]:
    return {
        "strategy_family": "adaptive_context",
        "base_condition": BASE_CONDITION,
        "condition": condition,
        "retrieval_unit": retrieval_unit,
        "expansion_policy": expansion_policy,
        "rerank_stage": rerank_stage,
        "packing_budget_words": packing_budget_words,
        "candidate_pool_k": candidate_pool_k,
        "final_top_k": final_top_k,
        "reranker_model": CROSS_ENCODER_MODEL_NAME,
        "context_order": "cross_encoder_score",
        "prompt_variant": "question_type_router",
        "question_type": question_type_from_text(question),
        "retrieved_context_count": len(contexts),
        "context_word_count": context_word_count(contexts),
        "avg_context_score": mean(context.score for context in contexts) if contexts else 0.0,
        "max_context_score": max((context.score for context in contexts), default=0.0),
        "prompt_word_count": len(prompt.split()),
    }


def fixed_baseline_contexts(question: str, chunks160: list[Chunk]) -> tuple[list[RetrievedContext], dict[str, Any]]:
    pool = retrieve_dense(question, chunks160, 20, model_name=DENSE_MODEL_NAME)
    contexts = cross_rerank_top(question, pool, 8, "dense_pool20_cross")
    return contexts, {
        "retrieval_unit": "fixed_chunk",
        "expansion_policy": "none",
        "rerank_stage": "after_retrieval",
        "packing_budget_words": 0,
        "candidate_pool_k": 20,
        "final_top_k": 8,
    }


def small_to_large_contexts(example: QAExample, condition: str) -> tuple[list[RetrievedContext], dict[str, Any]]:
    pool_k = 30 if "pool30" in condition else 20
    expand_tokens = 240 if "expand240" in condition else 160
    budget = 0
    if "budget600" in condition:
        budget = 600
    if "budget800" in condition:
        budget = 800

    small_chunks = chunk_paper(example.paper, chunk_size_tokens=80, chunk_overlap_tokens=20)
    pool = retrieve_dense(example.question, small_chunks, pool_k, model_name=DENSE_MODEL_NAME)

    if condition == "small80_expand160_cross_after_top8_router":
        expanded_pool = dedupe_contexts_by_text([
            expand_context_to_token_window(example, context, expand_tokens, "small80_expand160")
            for context in pool
        ])
        contexts = cross_rerank_top(example.question, expanded_pool, 8, "small80_expand160_cross_after")
        rerank_stage = "after_expansion"
    elif condition == "small80_cross_expand160_cross_top8_router":
        first_pass = cross_rerank_top(example.question, pool, min(16, len(pool)), "small80_cross_before")
        expanded_pool = dedupe_contexts_by_text([
            expand_context_to_token_window(example, context, expand_tokens, "small80_cross_expand160")
            for context in first_pass
        ])
        contexts = cross_rerank_top(example.question, expanded_pool, 8, "small80_cross_expand160_cross")
        rerank_stage = "before_and_after"
    else:
        first_pass = cross_rerank_top(example.question, pool, 8 if budget == 0 else min(14, len(pool)), "small80_cross_before")
        expanded = dedupe_contexts_by_text([
            expand_context_to_token_window(example, context, expand_tokens, f"small80_expand{expand_tokens}")
            for context in first_pass
        ])
        contexts = with_new_ranks(expanded[:8 if budget == 0 else len(expanded)], f"small80_cross_expand{expand_tokens}")
        rerank_stage = "before_expansion"

    if budget:
        contexts = pack_context_budget(contexts, budget, f"budget{budget}")

    return contexts, {
        "retrieval_unit": "small_chunk",
        "expansion_policy": "budget_pack" if budget else "parent_chunk",
        "rerank_stage": rerank_stage,
        "packing_budget_words": budget,
        "candidate_pool_k": pool_k,
        "final_top_k": len(contexts),
    }


def sentence_contexts(example: QAExample, condition: str) -> tuple[list[RetrievedContext], dict[str, Any]]:
    pool_k = 50 if "pool50" in condition else 30
    budget = 700 if "budget700" in condition else 0
    sentence_chunks, sentence_metadata = build_sentence_chunks(example.paper)
    pool = retrieve_dense(example.question, sentence_chunks, pool_k, model_name=DENSE_MODEL_NAME)
    first_pass = cross_rerank_top(example.question, pool, 8 if budget == 0 else min(16, len(pool)), "sentence_cross")

    if "paragraph" in condition:
        expanded = [
            expand_sentence_to_paragraph(example, context, "sentence_to_paragraph")
            for context in first_pass
        ]
        expansion_policy = "paragraph_window"
    else:
        radius = 2 if "pm2" in condition else 1
        expanded = [
            expand_sentence_window(example, context, sentence_metadata, radius, f"sentence_window_pm{radius}")
            for context in first_pass
        ]
        expansion_policy = "budget_pack" if budget else "sentence_window"

    contexts = with_new_ranks(dedupe_contexts_by_text(expanded)[:8 if budget == 0 else len(expanded)], "sentence_expanded")
    if budget:
        contexts = pack_context_budget(contexts, budget, f"sentence_budget{budget}")

    return contexts, {
        "retrieval_unit": "sentence",
        "expansion_policy": expansion_policy,
        "rerank_stage": "before_expansion",
        "packing_budget_words": budget,
        "candidate_pool_k": pool_k,
        "final_top_k": len(contexts),
    }


def paragraph_contexts(example: QAExample, condition: str) -> tuple[list[RetrievedContext], dict[str, Any]]:
    pool_k = 30 if "pool30" in condition else 20
    paragraph_chunks = build_paragraph_chunks(example.paper)
    pool = retrieve_dense(example.question, paragraph_chunks, pool_k, model_name=DENSE_MODEL_NAME)
    contexts = cross_rerank_top(example.question, pool, 8, "paragraph_cross")
    return contexts, {
        "retrieval_unit": "paragraph",
        "expansion_policy": "none",
        "rerank_stage": "after_retrieval",
        "packing_budget_words": 0,
        "candidate_pool_k": pool_k,
        "final_top_k": 8,
    }


def adaptive_contexts_for_condition(example: QAExample, condition: str) -> tuple[list[RetrievedContext], dict[str, Any]]:
    chunks160 = chunk_paper(example.paper, chunk_size_tokens=160, chunk_overlap_tokens=30)
    if condition == "dense_pool20_cross_top8_router":
        return fixed_baseline_contexts(example.question, chunks160)
    if condition.startswith("small80_"):
        return small_to_large_contexts(example, condition)
    if condition.startswith("sentence_"):
        return sentence_contexts(example, condition)
    if condition.startswith("paragraph_"):
        return paragraph_contexts(example, condition)
    raise ValueError(condition)


def run_adaptive_condition(example: QAExample, condition: str) -> dict[str, Any]:
    contexts, config = adaptive_contexts_for_condition(example, condition)
    prompt = build_question_type_router_prompt(example.question, contexts)
    raw_prediction = generate_text(prompt)
    prediction = trim_first_line(raw_prediction)
    context_texts = [context.chunk.text for context in contexts]
    trace = adaptive_trace(
        condition=condition,
        question=example.question,
        contexts=contexts,
        prompt=prompt,
        retrieval_unit=config["retrieval_unit"],
        expansion_policy=config["expansion_policy"],
        rerank_stage=config["rerank_stage"],
        packing_budget_words=config["packing_budget_words"],
        candidate_pool_k=config["candidate_pool_k"],
        final_top_k=config["final_top_k"],
    )
    return {
        "example_id": example.example_id,
        "condition": condition,
        "question": example.question,
        "gold_answers": example.gold_texts(),
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "metrics": score_prediction(prediction, example, context_texts),
        "retrieved_contexts": [context.as_dict() for context in contexts],
        "ablation_trace": trace,
    }


def summarize_adaptive(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for condition in sorted({record["condition"] for record in records}):
        condition_records = [record for record in records if record["condition"] == condition]
        row = {"condition": condition, "count": float(len(condition_records))}
        for metric in ["token_f1", "exact_match", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            row[metric] = mean(record["metrics"][metric] for record in condition_records)
        for diagnostic in [
            "candidate_pool_k",
            "final_top_k",
            "packing_budget_words",
            "retrieved_context_count",
            "context_word_count",
            "avg_context_score",
            "max_context_score",
            "prompt_word_count",
        ]:
            row[diagnostic] = mean(float(record["ablation_trace"][diagnostic]) for record in condition_records)
        first_trace = condition_records[0]["ablation_trace"]
        for label in [
            "strategy_family",
            "base_condition",
            "retrieval_unit",
            "expansion_policy",
            "rerank_stage",
            "reranker_model",
            "context_order",
            "prompt_variant",
        ]:
            row[label] = first_trace[label]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["token_f1", "context_recall"], ascending=False)


def trace_frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        trace = record["ablation_trace"]
        rows.append({
            "example_id": record["example_id"],
            "condition": record["condition"],
            "question_type": trace["question_type"],
            "retrieval_unit": trace["retrieval_unit"],
            "expansion_policy": trace["expansion_policy"],
            "rerank_stage": trace["rerank_stage"],
            "candidate_pool_k": trace["candidate_pool_k"],
            "final_top_k": trace["final_top_k"],
            "packing_budget_words": trace["packing_budget_words"],
            "context_word_count": trace["context_word_count"],
            "prompt_word_count": trace["prompt_word_count"],
            "token_f1": record["metrics"]["token_f1"],
            "exact_match": record["metrics"]["exact_match"],
            "context_precision": record["metrics"]["context_precision"],
            "context_recall": record["metrics"]["context_recall"],
            "prediction": record["prediction"],
            "gold_answers": record["gold_answers"],
        })
    return pd.DataFrame(rows)


def display_adaptive_outputs(records: list[dict[str, Any]]) -> pd.DataFrame:
    summary = summarize_adaptive(records)
    print("ADAPTIVE_CHUNK_FLEXIBLE_CONTEXT_FINAL_METRICS")
    display(summary)

    print("BEST_BY_STRATEGY")
    display(
        summary.sort_values(["retrieval_unit", "expansion_policy", "token_f1", "context_recall"], ascending=[True, True, False, False])
        .groupby(["retrieval_unit", "expansion_policy"], as_index=False)
        .head(2)
        .sort_values(["token_f1", "context_recall"], ascending=False)
    )

    trace_frame = trace_frame_from_records(records)
    print("QUESTION_TYPE_SUMMARY")
    display(
        trace_frame.groupby(["condition", "question_type"], as_index=False)
        .agg(
            count=("example_id", "count"),
            token_f1=("token_f1", "mean"),
            exact_match=("exact_match", "mean"),
            context_precision=("context_precision", "mean"),
            context_recall=("context_recall", "mean"),
        )
        .sort_values(["condition", "question_type"])
    )

    print("CONTEXT_BUDGET_WORD_COUNT_SUMMARY")
    display(
        trace_frame.groupby(["condition", "retrieval_unit", "expansion_policy", "packing_budget_words"], as_index=False)
        .agg(
            context_word_count=("context_word_count", "mean"),
            prompt_word_count=("prompt_word_count", "mean"),
            final_top_k=("final_top_k", "mean"),
            token_f1=("token_f1", "mean"),
            context_recall=("context_recall", "mean"),
        )
        .sort_values(["token_f1", "context_recall"], ascending=False)
    )

    print("CSV_READY_ROWS")
    display(summary[[
        "condition",
        "count",
        "token_f1",
        "exact_match",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "candidate_pool_k",
        "final_top_k",
        "context_word_count",
        "retrieval_unit",
        "expansion_policy",
        "rerank_stage",
        "packing_budget_words",
        "reranker_model",
        "prompt_variant",
    ]])

    print("SAMPLE_RECORD_PREVIEW")
    display(trace_frame.head(80))
    return summary
'''


RUN = r'''random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

examples = load_qasper_examples(SPLIT, limit=LIMIT)
print(f"Loaded {len(examples)} Qasper QA examples")
print(f"Adaptive chunk/flexible context conditions: {len(ADAPTIVE_CONDITIONS)}")
print(f"Expected records: {len(examples) * len(ADAPTIVE_CONDITIONS)}")
print("Baseline to beat: dense_pool20_cross_top8_router token_f1=0.313624, exact_match=0.120000, context_recall=0.509286.")

records = []
for index, example in enumerate(examples, start=1):
    print(f"[{index}/{len(examples)}] {example.example_id}")
    for condition in ADAPTIVE_CONDITIONS:
        records.append(run_adaptive_condition(example, condition))

print(f"Final record count: {len(records)}")
summary = display_adaptive_outputs(records)
'''


def build_notebook() -> dict:
    return {
        "cells": [
            markdown(
                "# Qasper Adaptive Chunk / Flexible Context Ablation\n\n"
                "Standalone Kaggle notebook testing whether flexible context construction can improve the SLM tradeoff between evidence precision, surrounding logic, and noise burden."
            ),
            code(SETUP),
            markdown(
                "## Configuration and Utilities\n\n"
                "All conditions use validation split, LIMIT=50, e5-small-v2 dense retrieval, MiniLM cross-encoder rerank, Qwen 0.5B generation, and the question-type router prompt. "
                "Gold evidence and answers are used only for scoring diagnostics."
            ),
            code(CORE + "\n\n" + ADAPTIVE_CORE),
            markdown(
                "## Run Experiment\n\n"
                "Runs the realistic best baseline plus small-to-large retrieval, rerank-stage comparisons, sentence-centered windows, paragraph retrieval, and budget-aware packing."
            ),
            code(RUN),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
