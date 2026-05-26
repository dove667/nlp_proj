from __future__ import annotations

from collections import defaultdict


def score_prediction(prediction: str, answers: list[str], metric: str) -> float:
    pred = prediction.strip().lower()
    golds = [answer.strip().lower() for answer in answers]
    if metric == "accuracy":
        return float(any(gold == pred or gold in pred for gold in golds))
    if metric in {"f1", "rouge_l"}:
        raise NotImplementedError(f"Implement {metric} scoring for the prepared dataset.")
    raise ValueError(f"Unsupported metric: {metric}")


def summarize_methods(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["benchmark"],
            row["task"],
            int(row["context_length"]),
            row["metric"],
        )
        grouped[key].append(row)

    by_setting = []
    for (method, benchmark, task, context_length, metric), values in sorted(grouped.items()):
        by_setting.append(
            {
                "method": method,
                "benchmark": benchmark,
                "task": task,
                "context_length": context_length,
                "metric": metric,
                "num_cases": len(values),
                "score": sum(float(row["score"]) for row in values) / len(values),
                "mean_latency_seconds": sum(float(row["latency_seconds"]) for row in values) / len(values),
            }
        )
    return {"by_setting": by_setting, "method_gains": compute_method_gains(by_setting)}


def compute_method_gains(by_setting: list[dict]) -> list[dict]:
    baseline = {
        (row["benchmark"], row["task"], row["context_length"], row["metric"]): row
        for row in by_setting
        if row["method"] == "baseline"
    }

    gains = []
    for row in by_setting:
        key = (row["benchmark"], row["task"], row["context_length"], row["metric"])
        base = baseline.get(key)
        if base is None or row["method"] == "baseline":
            continue
        gains.append(
            {
                **{k: row[k] for k in ["method", "benchmark", "task", "context_length", "metric"]},
                "score_gain": row["score"] - base["score"],
                "latency_overhead_seconds": row["mean_latency_seconds"] - base["mean_latency_seconds"],
            }
        )
    return gains
