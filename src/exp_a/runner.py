from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from exp_a.data_loader import RetrievalDataLoader
from exp_a.evaluator import exact_or_contains_match, summarize_accuracy
from exp_a.model_adapter import build_model
from exp_a.schema import load_config


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    loader = RetrievalDataLoader()
    rows: list[dict] = []

    for model_spec in config.models:
        model = build_model(model_spec)
        for benchmark in config.benchmarks:
            for task in benchmark.tasks:
                for context_length in config.context_lengths:
                    cases = loader.iter_cases(
                        benchmark,
                        task=task,
                        context_length=context_length,
                        needle_positions=config.needle_positions,
                    )
                    for case in cases:
                        prediction = model.generate(case.prompt)
                        rows.append(
                            {
                                "experiment": config.experiment,
                                "model": model_spec.name,
                                "architecture": model_spec.architecture,
                                "benchmark": benchmark.name,
                                "task": task,
                                "case_id": case.case_id,
                                "context_length": case.context_length,
                                "needle_position": case.needle_position,
                                "prediction": prediction,
                                "answer": case.answer,
                                "correct": exact_or_contains_match(prediction, case.answer),
                                "metadata": case.metadata,
                            }
                        )

    summary = summarize_accuracy(rows, threshold=config.effective_context_threshold)
    write_jsonl(config.output_dir / "predictions.jsonl", rows)
    write_json(config.output_dir / "summary.json", summary)
    write_csv(config.output_dir / "by_setting.csv", summary["by_setting"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp A retrieval baseline.")
    parser.add_argument("--config", default="configs/exp_a_retrieval.yaml")
    args = parser.parse_args()
    run(args.config)
