from __future__ import annotations

from collections import defaultdict


def exact_or_contains_match(prediction: str, answer: str) -> bool:
    pred = prediction.strip().lower()
    gold = answer.strip().lower()
    return pred == gold or gold in pred


def summarize_accuracy(rows: list[dict], *, threshold: float) -> dict:
    grouped: dict[tuple[str, str, str, int], list[bool]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["benchmark"], row["task"], int(row["context_length"]))
        grouped[key].append(bool(row["correct"]))

    by_setting = []
    effective_lengths: dict[tuple[str, str, str], int] = defaultdict(int)
    for (model, benchmark, task, context_length), values in sorted(grouped.items()):
        accuracy = sum(values) / len(values)
        by_setting.append(
            {
                "model": model,
                "benchmark": benchmark,
                "task": task,
                "context_length": context_length,
                "num_cases": len(values),
                "accuracy": accuracy,
            }
        )
        if accuracy >= threshold:
            effective_lengths[(model, benchmark, task)] = max(
                effective_lengths[(model, benchmark, task)], context_length
            )

    return {
        "by_setting": by_setting,
        "effective_context_lengths": [
            {
                "model": model,
                "benchmark": benchmark,
                "task": task,
                "effective_context_length": length,
            }
            for (model, benchmark, task), length in sorted(effective_lengths.items())
        ],
    }
