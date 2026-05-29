#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def edit_similarity(pred: str, gold: str) -> float:
    # LongBench official metric for repobench-p
    pred, gold = pred.strip(), gold.strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    # Levenshtein distance via DP
    m, n = len(pred), len(gold)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == gold[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return 1.0 - dp[n] / max(m, n)


def task_metric(task: str) -> str:
    if task in {"hotpotqa", "qasper"}:
        return "qa_f1"
    if task == "gov_report":
        return "rouge_l_f1"
    if task == "repobench-p":
        return "edit_similarity"
    raise ValueError(f"Unknown task: {task}")


def score_example(task: str, pred: str, outputs: List[str]) -> float:
    if task in {"hotpotqa", "qasper"}:
        return 100.0 * max((qa_f1_score(pred, gold) for gold in outputs), default=0.0)
    if task == "gov_report":
        return 100.0 * max((rouge_l_f1(pred, gold) for gold in outputs), default=0.0)
    if task == "repobench-p":
        return 100.0 * max((edit_similarity(pred, gold) for gold in outputs), default=0.0)
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
    parser = argparse.ArgumentParser(description="Analyze Exp B LongBench predictions.")
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--longbench_tasks", nargs="+", default=LONGBENCH_TASKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)

    summary_rows: List[Dict[str, Any]] = []

    for task in args.longbench_tasks:
        pred_file = pred_root / f"{task}.pred.jsonl"
        if not pred_file.exists():
            print(f"[SKIP] Missing prediction file: {pred_file}")
            continue
        rows = read_jsonl(pred_file)
        summary = summarize_rows(task, rows)
        summary.update(
            {
                "benchmark": "longbench",
                "task": task,
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
