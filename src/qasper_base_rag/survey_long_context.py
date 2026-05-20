from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from .data import document_word_count, extract_qa_examples, load_qasper


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = round((len(values) - 1) * ratio)
    return values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Survey whether Qasper papers are long-context enough.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[1000, 3000, 5000, 8000, 12000],
        help="Document word-count thresholds used to report long-context coverage.",
    )
    parser.add_argument("--output", default="outputs/qasper_long_context_survey.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_qasper(args.split)

    doc_lengths: list[int] = []
    qa_counts: list[int] = []
    examples_by_threshold = {threshold: 0 for threshold in args.thresholds}
    docs_by_threshold = {threshold: 0 for threshold in args.thresholds}

    for record in dataset:
        word_count = document_word_count(record)
        qa_count = len(extract_qa_examples(record))
        doc_lengths.append(word_count)
        qa_counts.append(qa_count)
        for threshold in args.thresholds:
            if word_count >= threshold:
                docs_by_threshold[threshold] += 1
                examples_by_threshold[threshold] += qa_count

    sorted_lengths = sorted(doc_lengths)
    total_docs = len(doc_lengths)
    total_examples = sum(qa_counts)
    summary = {
        "split": args.split,
        "documents": total_docs,
        "qa_examples": total_examples,
        "word_count": {
            "min": min(sorted_lengths) if sorted_lengths else 0,
            "p25": percentile(sorted_lengths, 0.25),
            "median": int(median(sorted_lengths)) if sorted_lengths else 0,
            "mean": mean(sorted_lengths) if sorted_lengths else 0.0,
            "p75": percentile(sorted_lengths, 0.75),
            "p90": percentile(sorted_lengths, 0.90),
            "p95": percentile(sorted_lengths, 0.95),
            "max": max(sorted_lengths) if sorted_lengths else 0,
        },
        "thresholds": {
            str(threshold): {
                "documents": docs_by_threshold[threshold],
                "document_rate": docs_by_threshold[threshold] / total_docs if total_docs else 0.0,
                "qa_examples": examples_by_threshold[threshold],
                "qa_example_rate": examples_by_threshold[threshold] / total_examples if total_examples else 0.0,
            }
            for threshold in args.thresholds
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"split={summary['split']}")
    print(f"documents={summary['documents']}")
    print(f"qa_examples={summary['qa_examples']}")
    for key, value in summary["word_count"].items():
        if isinstance(value, float):
            print(f"word_count_{key}={value:.2f}")
        else:
            print(f"word_count_{key}={value}")
    for threshold, stats in summary["thresholds"].items():
        print(
            f"threshold_{threshold}: docs={stats['documents']} "
            f"({stats['document_rate']:.1%}), qa={stats['qa_examples']} ({stats['qa_example_rate']:.1%})"
        )
    print(f"summary={output_path}")


if __name__ == "__main__":
    main()

