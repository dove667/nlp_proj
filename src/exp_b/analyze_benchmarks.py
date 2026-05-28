#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RULER_TASKS = ["vt", "cwe", "fwe"]
LONGBENCH_TASKS = ["hotpotqa", "qasper", "gov_report", "repobench-p"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


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


def normalize_qa_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def qa_f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_qa_answer(pred).split()
    gold_tokens = normalize_qa_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    dp = [[0] * (len(gold_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, p in enumerate(pred_tokens, start=1):
        for j, g in enumerate(gold_tokens, start=1):
            if p == g:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def code_exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def task_metric(task: str) -> str:
    if task in RULER_TASKS:
        return "accuracy"
    if task in {"hotpotqa", "qasper"}:
        return "qa_f1"
    if task == "gov_report":
        return "rouge_l_f1"
    if task == "repobench-p":
        return "exact_match"
    raise ValueError(f"Unknown task: {task}")


def score_example(task: str, pred: str, outputs: List[str]) -> float:
    if task in RULER_TASKS:
        return ruler_score(pred, outputs)
    if task in {"hotpotqa", "qasper"}:
        return 100.0 * max((qa_f1_score(pred, gold) for gold in outputs), default=0.0)
    if task == "gov_report":
        return 100.0 * max((rouge_l_f1(pred, gold) for gold in outputs), default=0.0)
    if task == "repobench-p":
        return 100.0 * max((code_exact_match(pred, gold) for gold in outputs), default=0.0)
    raise ValueError(f"Unknown task: {task}")


def summarize_rows(task: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [score_example(task, str(r.get("pred", "")), get_outputs(r)) for r in rows]
    total_input_tokens = sum(int(r.get("input_tokens", 0) or 0) for r in rows)
    total_generated_tokens = sum(int(r.get("generated_tokens", 0) or 0) for r in rows)
    total_generation_seconds = sum(float(r.get("generation_seconds", 0.0) or 0.0) for r in rows)
    null_predictions = sum(1 for r in rows if not str(r.get("pred", "")).strip())
    return {
        "metric": task_metric(task),
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
    parser = argparse.ArgumentParser(description="Analyze Exp B predictions for RULER reasoning and LongBench.")
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--ruler_lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--ruler_tasks", nargs="+", default=RULER_TASKS)
    parser.add_argument("--longbench_lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--longbench_tasks", nargs="+", default=LONGBENCH_TASKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)

    summary_rows: List[Dict[str, Any]] = []
    decay_rows: List[Dict[str, Any]] = []

    groups = [
        ("ruler_reasoning", args.ruler_lengths, args.ruler_tasks),
        ("longbench", args.longbench_lengths, args.longbench_tasks),
    ]

    for benchmark, lengths, tasks in groups:
        for task in tasks:
            per_task = []
            for length in lengths:
                pred_file = pred_root / benchmark / str(length) / f"{task}.pred.jsonl"
                if not pred_file.exists():
                    print(f"[SKIP] Missing prediction file: {pred_file}")
                    continue
                rows = read_jsonl(pred_file)
                summary = summarize_rows(task, rows)
                summary.update(
                    {
                        "benchmark": benchmark,
                        "task": task,
                        "length": length,
                        "pred_file": str(pred_file),
                    }
                )
                summary_rows.append(summary)
                per_task.append(summary)

            per_task.sort(key=lambda x: x["length"])
            if not per_task:
                continue
            base = per_task[0]
            for current in per_task[1:]:
                delta = current["score"] - base["score"]
                denom = current["length"] - base["length"]
                slope_per_1k = delta / denom * 1000 if denom else math.nan
                decay_rows.append(
                    {
                        "benchmark": benchmark,
                        "task": task,
                        "metric": current["metric"],
                        "from_length": base["length"],
                        "to_length": current["length"],
                        "from_score": base["score"],
                        "to_score": current["score"],
                        "delta_score": delta,
                        "slope_per_1k_tokens": slope_per_1k,
                    }
                )

    summary_path = pred_root / "summary.csv"
    decay_path = pred_root / "decay.csv"

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

    write_csv(
        decay_path,
        [
            "benchmark",
            "task",
            "metric",
            "from_length",
            "to_length",
            "from_score",
            "to_score",
            "delta_score",
            "slope_per_1k_tokens",
        ],
        decay_rows,
    )

    print(f"Saved summary to: {summary_path}")
    print(f"Saved decay table to: {decay_path}")


if __name__ == "__main__":
    main()
