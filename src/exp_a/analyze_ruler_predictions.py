import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_csv


def compute_position_bucket(row: Dict[str, Any], num_bins: int) -> Tuple[int, float]:
    pos = row.get("token_position_answer")
    total = row.get("length")
    if pos is None or total in (None, 0):
        return -1, -1.0
    rel = float(pos) / float(total)
    bucket = min(int(rel * num_bins), num_bins - 1)
    return bucket, rel


def summarize_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_examples = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    null_predictions = sum(1 for row in rows if not str(row.get("pred", "")).strip())
    total_input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in rows)
    total_generated_tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in rows)
    total_generation_seconds = sum(float(row.get("generation_seconds", 0.0) or 0.0) for row in rows)

    return {
        "num_examples": num_examples,
        "correct": correct,
        "score": 100.0 * correct / max(num_examples, 1),
        "null_predictions": null_predictions,
        "avg_input_tokens": total_input_tokens / max(num_examples, 1),
        "avg_generated_tokens": total_generated_tokens / max(num_examples, 1),
        "total_generation_seconds": total_generation_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize RULER prediction files with global metrics and position sensitivity buckets."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_1"])
    parser.add_argument("--num_bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)

    summary_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []

    for length in args.lengths:
        for task in args.tasks:
            data_file = data_root / str(length) / task / "test.jsonl"
            pred_file = pred_root / str(length) / f"{task}.pred.jsonl"
            if not pred_file.exists():
                print(f"[SKIP] Missing prediction file: {pred_file}")
                continue

            rows = read_jsonl(pred_file)
            summary = summarize_predictions(rows)
            summary.update(
                {
                    "length": length,
                    "task": task,
                    "data_file": str(data_file),
                    "pred_file": str(pred_file),
                }
            )
            summary_rows.append(summary)

            buckets: List[List[Tuple[float, Dict[str, Any]]]] = [[] for _ in range(args.num_bins)]
            invalid_rows = 0
            for row in rows:
                bucket_idx, rel_pos = compute_position_bucket(row, args.num_bins)
                if bucket_idx < 0:
                    invalid_rows += 1
                    continue
                buckets[bucket_idx].append((rel_pos, row))

            for bucket_idx, bucket_data in enumerate(buckets):
                bucket_correct = sum(1 for _, row in bucket_data if row.get("correct"))
                bucket_total = len(bucket_data)
                avg_rel_pos = sum(rel for rel, _ in bucket_data) / bucket_total if bucket_total else None
                bucket_rows.append(
                    {
                        "length": length,
                        "task": task,
                        "bucket_id": bucket_idx,
                        "bucket_start_pct": bucket_idx * 100.0 / args.num_bins,
                        "bucket_end_pct": (bucket_idx + 1) * 100.0 / args.num_bins,
                        "num_examples": bucket_total,
                        "correct": bucket_correct,
                        "score": 100.0 * bucket_correct / max(bucket_total, 1),
                        "avg_relative_position": avg_rel_pos,
                        "invalid_position_rows": invalid_rows,
                        "pred_file": str(pred_file),
                    }
                )

    summary_path = pred_root / "summary.csv"
    bucket_path = pred_root / "position_sensitivity_10bins.csv"

    write_csv(
        summary_path,
        [
            "length",
            "task",
            "score",
            "num_examples",
            "correct",
            "null_predictions",
            "avg_input_tokens",
            "avg_generated_tokens",
            "total_generation_seconds",
            "data_file",
            "pred_file",
        ],
        summary_rows,
    )

    write_csv(
        bucket_path,
        [
            "length",
            "task",
            "bucket_id",
            "bucket_start_pct",
            "bucket_end_pct",
            "num_examples",
            "correct",
            "score",
            "avg_relative_position",
            "invalid_position_rows",
            "pred_file",
        ],
        bucket_rows,
    )

    print(f"Saved global summary to: {summary_path}")
    print(f"Saved position summary to: {bucket_path}")


if __name__ == "__main__":
    main()
