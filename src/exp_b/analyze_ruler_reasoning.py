import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_csv


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\x00-\x1f]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_outputs(example: Dict[str, Any]) -> List[str]:
    outputs = example.get("outputs")
    if outputs is None:
        outputs = example.get("output")
    if outputs is None:
        outputs = example.get("answers")
    if outputs is None:
        return []
    if isinstance(outputs, list):
        return [str(x) for x in outputs]
    return [str(outputs)]


def answer_in_prediction(answer: str, pred: str) -> bool:
    answer_norm = normalize_text(answer)
    pred_norm = normalize_text(pred)
    if not answer_norm:
        return False
    if re.fullmatch(r"[0-9,\s.\-]+", answer_norm):
        pattern = r"(?<!\d)" + re.escape(answer_norm) + r"(?!\d)"
        return re.search(pattern, pred_norm) is not None
    return answer_norm in pred_norm


def ruler_score(pred: str, outputs: List[str]) -> float:
    return 100.0 if all(answer_in_prediction(ans, pred) for ans in outputs) else 0.0


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [ruler_score(str(r.get("pred", "")), get_outputs(r)) for r in rows]
    total_input_tokens = sum(int(r.get("input_tokens", 0) or 0) for r in rows)
    total_generated_tokens = sum(int(r.get("generated_tokens", 0) or 0) for r in rows)
    total_generation_seconds = sum(float(r.get("generation_seconds", 0.0) or 0.0) for r in rows)
    null_predictions = sum(1 for r in rows if not str(r.get("pred", "")).strip())
    return {
        "metric": "accuracy",
        "score": sum(scores) / max(len(scores), 1),
        "num_examples": len(rows),
        "null_predictions": null_predictions,
        "avg_input_tokens": total_input_tokens / max(len(rows), 1),
        "avg_generated_tokens": total_generated_tokens / max(len(rows), 1),
        "total_generation_seconds": total_generation_seconds,
        "min_example_score": min(scores) if scores else 0.0,
        "max_example_score": max(scores) if scores else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp B RULER reasoning predictions.")
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--ruler_lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--ruler_tasks", nargs="+", default=["vt", "cwe", "fwe"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)

    summary_rows: List[Dict[str, Any]] = []

    for task in args.ruler_tasks:
        for length in args.ruler_lengths:
            pred_file = pred_root / str(length) / f"{task}.pred.jsonl"
            if not pred_file.exists():
                print(f"[SKIP] Missing prediction file: {pred_file}")
                continue
            rows = read_jsonl(pred_file)
            summary = summarize_rows(rows)
            summary.update(
                {
                    "benchmark": "ruler_reasoning",
                    "task": task,
                    "length": length,
                    "pred_file": str(pred_file),
                }
            )
            summary_rows.append(summary)

    summary_path = pred_root / "summary.csv"
    write_csv(
        summary_path,
        [
            "benchmark",
            "task",
            "length",
            "metric",
            "score",
            "num_examples",
            "null_predictions",
            "avg_input_tokens",
            "avg_generated_tokens",
            "total_generation_seconds",
            "min_example_score",
            "max_example_score",
            "pred_file",
        ],
        summary_rows,
    )

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
