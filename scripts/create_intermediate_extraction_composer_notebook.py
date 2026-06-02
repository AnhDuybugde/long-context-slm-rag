"""Create a standalone intermediate extraction + rule/IR composer notebook."""

from __future__ import annotations

import json
from pathlib import Path

from create_adaptive_chunk_flexible_context_notebook import ADAPTIVE_CORE
from create_cache_memory_notebook import SETUP, code, markdown
from create_evidence_selected_notebook import CORE


NOTEBOOK_PATH = Path("notebooks/19-intermediate-extraction-rule-ir-composer.ipynb")


EXTRACTION_COMPOSER_CORE = r'''
BASE_CONDITION = "small80_pool30_cross_expand160_budget800_router"

COMPOSER_CONDITIONS = [
    "best_adaptive_direct_router",
    "best_adaptive_slm_extract_then_slm_answer",
    "best_adaptive_slm_extract_then_rule_compose",
    "best_adaptive_ir_extract_then_rule_compose",
    "best_adaptive_hybrid_extract_then_rule_compose",
    "dense_pool20_cross_slm_extract_then_rule_compose",
    "oracle_evidence_slm_extract_then_rule_compose",
]

NEGATION_RE = re.compile(r"\b(no|not|never|none|neither|cannot|can't|does not|do not|did not|without|fail|fails|failed|unanswerable|unknown)\b", re.I)
AFFIRMATION_RE = re.compile(r"\b(yes|supports|show|shows|demonstrate|demonstrates|use|uses|used|include|includes|included|is|are|was|were|can|could|does|do)\b", re.I)
QUESTION_STOPWORDS = {
    "what", "which", "who", "where", "when", "how", "why", "does", "do", "did", "is", "are", "was", "were",
    "can", "could", "will", "would", "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "with",
    "this", "that", "these", "those", "paper", "study", "authors", "author"
}


def answer_type_from_example(example: QAExample) -> str:
    question_type = question_type_from_text(example.question)
    if any(answer.unanswerable for answer in example.answers):
        return "unanswerable"
    if question_type == "yes_no":
        return "yes_no"
    if any(answer.extractive_spans for answer in example.answers):
        return "extractive"
    return "free_form"


def content_token_set(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in QUESTION_STOPWORDS and len(token) > 2}


def evidence_overlap_score(question: str, sentence: str, source_score: float = 0.0, rank: int = 1) -> float:
    q_tokens = content_token_set(question)
    s_tokens = content_token_set(sentence)
    lexical = len(q_tokens & s_tokens) / max(1, len(q_tokens))
    q_entities = set(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b|\b\d+(?:\.\d+)?%?\b", question))
    s_entities = set(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b|\b\d+(?:\.\d+)?%?\b", sentence))
    entity = len(q_entities & s_entities) / max(1, len(q_entities)) if q_entities else 0.0
    number_bonus = 0.08 if re.search(r"\d", question) and re.search(r"\d", sentence) else 0.0
    rank_bonus = 1.0 / max(2.0, rank + 1.0)
    return 0.52 * lexical + 0.18 * entity + 0.18 * float(source_score) + 0.07 * rank_bonus + number_bonus


def evidence_precision_recall(selected_texts: list[str], gold_texts: list[str]) -> tuple[float, float]:
    selected = [normalize_answer(text) for text in selected_texts if normalize_answer(text)]
    gold = [normalize_answer(text) for text in gold_texts if normalize_answer(text)]
    if not selected:
        precision = 1.0 if not gold else 0.0
    else:
        selected_hits = 0
        for selected_text in selected:
            if any(selected_text in gold_text or gold_text in selected_text for gold_text in gold):
                selected_hits += 1
        precision = selected_hits / max(1, len(selected))
    if not gold:
        recall = 1.0
    else:
        gold_hits = 0
        for gold_text in gold:
            if any(selected_text in gold_text or gold_text in selected_text for selected_text in selected):
                gold_hits += 1
        recall = gold_hits / max(1, len(gold))
    return float(precision), float(recall)


def evidence_dict(text: str, score: float, source: str, context_rank: int = 0, sentence_index: int = 0) -> dict[str, Any]:
    return {
        "text": " ".join(str(text or "").split()).strip(),
        "score": float(score),
        "source": source,
        "context_rank": int(context_rank),
        "sentence_index": int(sentence_index),
    }


def split_composer_sentences(text: str) -> list[str]:
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


def ir_extract_evidence(question: str, contexts: list[RetrievedContext], max_evidence: int = 5) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        for sentence_index, sentence in enumerate(split_composer_sentences(context.chunk.text), start=1):
            score = evidence_overlap_score(question, sentence, context.score, context.rank)
            rows.append(evidence_dict(sentence, score, "ir_rule", context.rank, sentence_index))
    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = []
    seen = set()
    per_context_counts: dict[int, int] = {}
    for row in rows:
        key = normalize_answer(row["text"])
        if not key or key in seen:
            continue
        if row["score"] < 0.08 and selected:
            continue
        context_rank = row["context_rank"]
        if per_context_counts.get(context_rank, 0) >= 2:
            continue
        selected.append(row)
        seen.add(key)
        per_context_counts[context_rank] = per_context_counts.get(context_rank, 0) + 1
        if len(selected) >= max_evidence:
            break
    return selected


def build_slm_extraction_prompt(example: QAExample, contexts: list[RetrievedContext]) -> str:
    return (
        "Select the minimal evidence needed to answer the question. "
        "Return at most 5 short evidence lines copied or closely paraphrased from the context. "
        "Do not answer the question yet. If no evidence supports an answer, return Unanswerable.\n\n"
        f"Question type: {answer_type_from_example(example)}\n\n"
        f"Context:\n{context_block(contexts)}\n\n"
        f"Question: {example.question}\n"
        "Evidence:\n"
        "1."
    )


def parse_slm_evidence(raw_text: str, question: str, contexts: list[RetrievedContext], max_evidence: int = 5) -> list[dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return []
    if re.search(r"\bunanswerable\b", text, flags=re.I):
        return [evidence_dict("Unanswerable", 0.0, "slm", 0, 0)]
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)]|evidence\s*\d*\s*:)\s*", "", line, flags=re.I).strip()
        cleaned = cleaned.strip(" \"'")
        if not cleaned or cleaned.lower().startswith(("question:", "answer:", "final answer")):
            continue
        lines.append(cleaned)
    if not lines:
        lines = split_composer_sentences(text)
    rows = []
    for index, line in enumerate(lines[:max_evidence], start=1):
        score = evidence_overlap_score(question, line, 0.0, index)
        rows.append(evidence_dict(line, score, "slm", 0, index))
    if rows:
        return rows
    return ir_extract_evidence(question, contexts, max_evidence=max_evidence)


def slm_extract_evidence(example: QAExample, contexts: list[RetrievedContext], max_evidence: int = 5) -> tuple[list[dict[str, Any]], str]:
    prompt = build_slm_extraction_prompt(example, contexts)
    raw = generate_text(prompt)
    return parse_slm_evidence(raw, example.question, contexts, max_evidence=max_evidence), raw


def merge_evidence_lists(first: list[dict[str, Any]], second: list[dict[str, Any]], max_evidence: int = 5) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for row in first + second:
        key = normalize_answer(row["text"])
        if not key or key in seen:
            continue
        merged.append(row)
        seen.add(key)
        if len(merged) >= max_evidence:
            break
    return merged


def evidence_contexts_from_rows(example: QAExample, evidence_rows: list[dict[str, Any]], source: str) -> list[RetrievedContext]:
    contexts = []
    for index, row in enumerate(evidence_rows, start=1):
        text = row["text"]
        chunk = Chunk(
            paper_id=example.paper_id,
            chunk_id=f"{example.paper_id}:{example.question_id}:{source}:{index}",
            section=f"{source}_evidence",
            text=text,
            title=example.paper.title,
            start_token=0,
            end_token=len(text.split()),
        )
        contexts.append(RetrievedContext(chunk, float(row.get("score", 0.0)), index, source))
    return contexts


def build_slm_answer_from_evidence_prompt(example: QAExample, evidence_rows: list[dict[str, Any]]) -> str:
    evidence_text = "\n".join(f"[E{index}] {row['text']}" for index, row in enumerate(evidence_rows, start=1)) or "[No evidence]"
    answer_type = answer_type_from_example(example)
    if answer_type == "yes_no":
        instruction = "Return only Yes, No, or Unanswerable."
    elif answer_type == "extractive":
        instruction = "Return only the shortest answer phrase supported by the evidence."
    else:
        instruction = "Return one short answer sentence supported by the evidence."
    return (
        "Answer using only the extracted evidence. "
        f"{instruction}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {example.question}\n"
        "Answer:"
    )


def yes_no_rule(question: str, evidence_rows: list[dict[str, Any]]) -> str:
    joined = " ".join(row["text"] for row in evidence_rows)
    if not joined.strip() or normalize_answer(joined) == "unanswerable":
        return "Unanswerable"
    if NEGATION_RE.search(joined):
        return "No"
    if AFFIRMATION_RE.search(joined) or content_token_set(question) & content_token_set(joined):
        return "Yes"
    return "Unanswerable"


def shortest_informative_span(question: str, evidence_rows: list[dict[str, Any]], max_words: int = 18) -> str:
    q_tokens = content_token_set(question)
    candidates = []
    for row in evidence_rows:
        text = row["text"]
        for sentence in split_composer_sentences(text) or [text]:
            words = sentence.split()
            if not words:
                continue
            score = evidence_overlap_score(question, sentence, row.get("score", 0.0), row.get("context_rank", 1))
            entity_or_number = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s+[A-Z][A-Za-z0-9-]{2,}){0,4}\b|\b\d+(?:\.\d+)?%?\b", sentence)
            if entity_or_number:
                span = " ".join(entity_or_number[:4])
            else:
                filtered = [word.strip(" ,.;:()[]") for word in words if normalize_answer(word) not in q_tokens]
                span = " ".join(filtered[:max_words]) or " ".join(words[:max_words])
            candidates.append((score, len(span.split()), span.strip(" ,.;:")))
    candidates = [(score, length, span) for score, length, span in candidates if span]
    if not candidates:
        return "Unanswerable"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def rule_compose_answer(example: QAExample, evidence_rows: list[dict[str, Any]]) -> str:
    answer_type = answer_type_from_example(example)
    filtered = [row for row in evidence_rows if normalize_answer(row["text"]) != "unanswerable"]
    if not filtered:
        return "Unanswerable"
    max_score = max((row.get("score", 0.0) for row in filtered), default=0.0)
    if max_score < 0.07 and answer_type != "free_form":
        return "Unanswerable"
    if answer_type == "yes_no":
        return yes_no_rule(example.question, filtered)
    if answer_type == "extractive":
        return shortest_informative_span(example.question, filtered)
    facts = []
    seen = set()
    for row in sorted(filtered, key=lambda item: item.get("score", 0.0), reverse=True):
        sentence = (split_composer_sentences(row["text"]) or [row["text"]])[0]
        sentence = sentence.strip()
        key = normalize_answer(sentence)
        if not key or key in seen:
            continue
        facts.append(sentence)
        seen.add(key)
        if len(facts) >= 2:
            break
    return " ".join(facts).strip() or shortest_informative_span(example.question, filtered, max_words=24)


def oracle_evidence_contexts(example: QAExample) -> list[RetrievedContext]:
    contexts = []
    for index, text in enumerate(example.evidence_texts(), start=1):
        chunk = Chunk(
            paper_id=example.paper_id,
            chunk_id=f"{example.paper_id}:oracle_evidence:{example.question_id}:{index}",
            section="gold_evidence",
            text=text,
            title=example.paper.title,
            start_token=0,
            end_token=len(text.split()),
        )
        contexts.append(RetrievedContext(chunk, 1.0, index, "oracle_evidence"))
    return contexts or []


def contexts_for_composer_condition(example: QAExample, condition: str) -> tuple[list[RetrievedContext], str]:
    if condition.startswith("dense_pool20_cross"):
        contexts, _ = adaptive_contexts_for_condition(example, "dense_pool20_cross_top8_router")
        return contexts, "dense_pool20_cross_top8_router"
    if condition.startswith("oracle_evidence"):
        return oracle_evidence_contexts(example), "oracle_evidence"
    contexts, _ = adaptive_contexts_for_condition(example, "small80_pool30_cross_expand160_budget800_router")
    return contexts, "small80_pool30_cross_expand160_budget800_router"


def extraction_for_condition(example: QAExample, condition: str, contexts: list[RetrievedContext]) -> tuple[list[dict[str, Any]], str, str]:
    if condition == "best_adaptive_ir_extract_then_rule_compose":
        return ir_extract_evidence(example.question, contexts), "", "ir_rule"
    if condition == "best_adaptive_hybrid_extract_then_rule_compose":
        slm_rows, raw = slm_extract_evidence(example, contexts)
        ir_rows = ir_extract_evidence(example.question, contexts)
        return merge_evidence_lists(slm_rows, ir_rows), raw, "hybrid"
    rows, raw = slm_extract_evidence(example, contexts)
    return rows, raw, "slm"


def error_tag_for_record(record: dict[str, Any]) -> str:
    trace = record["ablation_trace"]
    token_f1 = record["metrics"]["token_f1"]
    if trace["extraction_recall"] < 0.25:
        return "low_extraction_recall"
    if trace["extraction_precision"] < 0.25:
        return "noisy_extraction"
    if token_f1 < 0.25:
        return "composition_or_wording_error"
    return "ok_or_partial"


def run_composer_condition(example: QAExample, condition: str) -> dict[str, Any]:
    contexts, context_source = contexts_for_composer_condition(example, condition)
    raw_extraction = ""
    evidence_rows: list[dict[str, Any]] = []
    composer_source = "direct_slm"

    if condition == "best_adaptive_direct_router":
        prompt = build_question_type_router_prompt(example.question, contexts)
        raw_prediction = generate_text(prompt)
        prediction = trim_first_line(raw_prediction)
        evidence_rows = ir_extract_evidence(example.question, contexts)
        selected_contexts = contexts
        composer_source = "direct_slm"
    else:
        evidence_rows, raw_extraction, extractor_source = extraction_for_condition(example, condition, contexts)
        selected_contexts = evidence_contexts_from_rows(example, evidence_rows, extractor_source)
        if condition == "best_adaptive_slm_extract_then_slm_answer":
            prompt = build_slm_answer_from_evidence_prompt(example, evidence_rows)
            raw_prediction = generate_text(prompt)
            prediction = trim_first_line(raw_prediction)
            composer_source = "slm"
        else:
            prompt = ""
            raw_prediction = rule_compose_answer(example, evidence_rows)
            prediction = raw_prediction
            composer_source = extractor_source if extractor_source != "slm" else "slm_rule"

    context_texts = [context.chunk.text for context in selected_contexts]
    evidence_texts = [row["text"] for row in evidence_rows]
    extraction_precision, extraction_recall = evidence_precision_recall(evidence_texts, example.evidence_texts())
    metrics = score_prediction(prediction, example, context_texts)
    trace = {
        "strategy_family": "intermediate_extraction_composer",
        "base_condition": BASE_CONDITION,
        "context_source": context_source,
        "answer_type": answer_type_from_example(example),
        "composer_source": composer_source,
        "selected_evidence_count": len(evidence_rows),
        "evidence_score": mean(row.get("score", 0.0) for row in evidence_rows) if evidence_rows else 0.0,
        "extraction_precision": extraction_precision,
        "extraction_recall": extraction_recall,
        "answer_token_f1": metrics["token_f1"],
        "exact_match": metrics["exact_match"],
        "context_word_count": sum(len(text.split()) for text in context_texts),
        "prompt_word_count": len(prompt.split()) if prompt else 0,
        "question_type": question_type_from_text(example.question),
    }
    record = {
        "example_id": example.example_id,
        "condition": condition,
        "question": example.question,
        "gold_answers": example.gold_texts(),
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "raw_extraction": raw_extraction,
        "selected_evidence": evidence_rows,
        "metrics": metrics,
        "retrieved_contexts": [context.as_dict() for context in selected_contexts],
        "ablation_trace": trace,
    }
    record["error_tag"] = error_tag_for_record(record)
    return record


def summarize_composer(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for condition in sorted({record["condition"] for record in records}):
        condition_records = [record for record in records if record["condition"] == condition]
        row = {"condition": condition, "count": float(len(condition_records))}
        for metric in ["token_f1", "exact_match", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            row[metric] = mean(record["metrics"][metric] for record in condition_records)
        for diagnostic in [
            "extraction_precision",
            "extraction_recall",
            "selected_evidence_count",
            "evidence_score",
            "context_word_count",
            "prompt_word_count",
        ]:
            row[diagnostic] = mean(float(record["ablation_trace"][diagnostic]) for record in condition_records)
        first_trace = condition_records[0]["ablation_trace"]
        for label in ["strategy_family", "base_condition", "context_source", "composer_source"]:
            row[label] = first_trace[label]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["token_f1", "extraction_recall"], ascending=False)


def composer_trace_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        trace = record["ablation_trace"]
        rows.append({
            "example_id": record["example_id"],
            "condition": record["condition"],
            "answer_type": trace["answer_type"],
            "question_type": trace["question_type"],
            "composer_source": trace["composer_source"],
            "selected_evidence_count": trace["selected_evidence_count"],
            "extraction_precision": trace["extraction_precision"],
            "extraction_recall": trace["extraction_recall"],
            "token_f1": record["metrics"]["token_f1"],
            "exact_match": record["metrics"]["exact_match"],
            "context_precision": record["metrics"]["context_precision"],
            "context_recall": record["metrics"]["context_recall"],
            "error_tag": record["error_tag"],
            "prediction": record["prediction"],
            "gold_answers": record["gold_answers"],
            "selected_evidence": [row["text"] for row in record["selected_evidence"]],
        })
    return pd.DataFrame(rows)


def display_composer_outputs(records: list[dict[str, Any]]) -> pd.DataFrame:
    summary = summarize_composer(records)
    print("INTERMEDIATE_EXTRACTION_COMPOSER_FINAL_METRICS")
    display(summary)

    print("EXTRACTION_PRECISION_RECALL_TABLE")
    display(summary[[
        "condition",
        "context_source",
        "composer_source",
        "extraction_precision",
        "extraction_recall",
        "selected_evidence_count",
        "token_f1",
        "exact_match",
    ]].sort_values(["extraction_recall", "token_f1"], ascending=False))

    trace_frame = composer_trace_frame(records)
    print("ANSWER_TYPE_SUMMARY")
    display(
        trace_frame.groupby(["condition", "answer_type"], as_index=False)
        .agg(
            count=("example_id", "count"),
            token_f1=("token_f1", "mean"),
            exact_match=("exact_match", "mean"),
            extraction_precision=("extraction_precision", "mean"),
            extraction_recall=("extraction_recall", "mean"),
        )
        .sort_values(["condition", "answer_type"])
    )

    print("DIRECT_VS_EXTRACT_VS_COMPOSE")
    display(
        summary.assign(
            route=summary["condition"].map(lambda value: "direct" if "direct" in value else "extract_then_slm" if "slm_answer" in value else "extract_then_rule")
        )[["route", "condition", "token_f1", "exact_match", "extraction_precision", "extraction_recall", "context_recall"]]
        .sort_values(["route", "token_f1"], ascending=[True, False])
    )

    print("PER_EXAMPLE_ERROR_TAGS")
    display(
        trace_frame.groupby(["condition", "error_tag"], as_index=False)
        .agg(count=("example_id", "count"), token_f1=("token_f1", "mean"))
        .sort_values(["condition", "count"], ascending=[True, False])
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
        "extraction_precision",
        "extraction_recall",
        "selected_evidence_count",
        "context_source",
        "composer_source",
    ]])

    print("SAMPLE_RECORD_PREVIEW")
    display(trace_frame.head(80))
    return summary
'''


RUN = r'''random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

examples = load_qasper_examples(SPLIT, limit=LIMIT)
print(f"Loaded {len(examples)} Qasper QA examples")
print(f"Intermediate extraction/composer conditions: {len(COMPOSER_CONDITIONS)}")
print(f"Expected records: {len(examples) * len(COMPOSER_CONDITIONS)}")
print("Best realistic reference: small80_pool30_cross_expand160_budget800_router token_f1=0.319529.")
print("Stable rerank reference: dense_pool20_cross_top8_router token_f1=0.313624.")
print("Oracle evidence-only router reference: token_f1=0.312332.")
print("Oracle extract-then-answer upper diagnostic: token_f1=0.510129.")

records = []
for index, example in enumerate(examples, start=1):
    print(f"[{index}/{len(examples)}] {example.example_id}")
    for condition in COMPOSER_CONDITIONS:
        records.append(run_composer_condition(example, condition))

print(f"Final record count: {len(records)}")
summary = display_composer_outputs(records)
'''


def build_notebook() -> dict:
    return {
        "cells": [
            markdown(
                "# Qasper Intermediate Extraction + Rule/IR Answer Composer\n\n"
                "Standalone diagnostic notebook to test whether Qwen 0.5B is bottlenecked by evidence selection, reasoning/synthesis, or answer wording."
            ),
            code(SETUP),
            markdown(
                "## Configuration and Utilities\n\n"
                "This notebook reuses the adaptive context builder from notebook 18 inline, then adds intermediate evidence extraction and deterministic answer composition diagnostics. "
                "Gold evidence and answers are used only for scoring."
            ),
            code(CORE + "\n\n" + ADAPTIVE_CORE + "\n\n" + EXTRACTION_COMPOSER_CORE),
            markdown(
                "## Run Experiment\n\n"
                "Runs direct adaptive baseline, SLM extraction to SLM answer, SLM/IR/hybrid extraction to rule composer, dense rerank extraction, and oracle-evidence extraction diagnostics."
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
