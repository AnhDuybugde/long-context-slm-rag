from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "independent_variants" / "qasper_long_context_survey_standalone.ipynb"


SURVEY_CELL = '''def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    return values[round((len(values) - 1) * ratio)]


def threshold_flags(word_count: int) -> dict[str, bool]:
    return {str(threshold): word_count >= threshold for threshold in THRESHOLDS}


def run_long_context_survey(split: str = SPLIT):
    dataset = load_qasper(split)
    lengths = []
    qa_counts = []
    document_rows = []
    docs_by_threshold = {threshold: 0 for threshold in THRESHOLDS}
    qa_by_threshold = {threshold: 0 for threshold in THRESHOLDS}

    for record in dataset:
        word_count = document_word_count(record)
        qa_examples = extract_qa_examples(record)
        qa_count = len(qa_examples)
        lengths.append(word_count)
        qa_counts.append(qa_count)
        document_rows.append(
            {
                "split": split,
                "doc_id": record.get("id", ""),
                "title": record.get("title", ""),
                "word_count": word_count,
                "qa_examples": qa_count,
                "thresholds": threshold_flags(word_count),
            }
        )
        for threshold in THRESHOLDS:
            if word_count >= threshold:
                docs_by_threshold[threshold] += 1
                qa_by_threshold[threshold] += qa_count

    lengths = sorted(lengths)
    total_docs = len(lengths)
    total_qa = sum(qa_counts)
    threshold_rows = [
        {
            "split": split,
            "threshold": threshold,
            "documents": docs_by_threshold[threshold],
            "document_rate": docs_by_threshold[threshold] / total_docs if total_docs else 0.0,
            "qa_examples": qa_by_threshold[threshold],
            "qa_example_rate": qa_by_threshold[threshold] / total_qa if total_qa else 0.0,
        }
        for threshold in THRESHOLDS
    ]
    summary = {
        "split": split,
        "documents": total_docs,
        "qa_examples": total_qa,
        "word_count": {
            "min": min(lengths),
            "p25": percentile(lengths, 0.25),
            "median": int(median(lengths)),
            "mean": mean(lengths),
            "p75": percentile(lengths, 0.75),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "thresholds": {
            str(row["threshold"]): {
                key: row[key]
                for key in ("documents", "document_rate", "qa_examples", "qa_example_rate")
            }
            for row in threshold_rows
        },
    }
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"qasper_long_context_survey_{split}.json"
    documents_path = output_dir / f"qasper_long_context_survey_{split}_documents.jsonl"
    thresholds_path = output_dir / f"qasper_long_context_survey_{split}_thresholds.jsonl"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with documents_path.open("w", encoding="utf-8") as file:
        for row in document_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\\n")
    with thresholds_path.open("w", encoding="utf-8") as file:
        for row in threshold_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\\n")

    summary["summary_path"] = str(summary_path)
    summary["documents_jsonl_path"] = str(documents_path)
    summary["thresholds_jsonl_path"] = str(thresholds_path)
    return summary
'''


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"][5]["source"] = SURVEY_CELL.splitlines(keepends=True)
    notebook["cells"][6]["source"] = [
        "summary = run_long_context_survey(SPLIT)\n",
        "summary\n",
    ]
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
