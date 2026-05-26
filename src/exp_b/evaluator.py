from __future__ import annotations

from collections import defaultdict


def score_prediction(prediction: str, answers: list[str], metric: str) -> float:
    normalized_prediction = prediction.strip().lower()
    normalized_answers = [answer.strip().lower() for answer in answers]

    if metric == "accuracy":
        return float(any(answer == normalized_prediction or answer in normalized_prediction for answer in normalized_answers))
    if metric in {"f1", "rouge_l"}:
        raise NotImplementedError(f"Implement {metric} scoring for the server dataset format.")
    raise ValueError(f"Unsupported metric: {metric}")


def summarize_scores(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["benchmark"],
            row["task"],
            int(row["context_length"]),
            row["metric"],
        )
        grouped[key].append(float(row["score"]))

    by_setting = []
    for (model, benchmark, task, context_length, metric), values in sorted(grouped.items()):
        by_setting.append(
            {
                "model": model,
                "benchmark": benchmark,
                "task": task,
                "context_length": context_length,
                "metric": metric,
                "num_cases": len(values),
                "score": sum(values) / len(values),
            }
        )

    return {
        "by_setting": by_setting,
        "accuracy_decay_slope": compute_decay_slopes(by_setting),
    }


def compute_decay_slopes(by_setting: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in by_setting:
        grouped[(row["model"], row["benchmark"], row["task"], row["metric"])].append(row)

    slopes = []
    for (model, benchmark, task, metric), values in grouped.items():
        values = sorted(values, key=lambda row: row["context_length"])
        if len(values) < 2:
            continue
        first, last = values[0], values[-1]
        delta_len = last["context_length"] - first["context_length"]
        slope = 0.0 if delta_len == 0 else (last["score"] - first["score"]) / delta_len
        slopes.append(
            {
                "model": model,
                "benchmark": benchmark,
                "task": task,
                "metric": metric,
                "from_context_length": first["context_length"],
                "to_context_length": last["context_length"],
                "slope": slope,
            }
        )
    return slopes
