from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from exp_b.data_loader import ReasoningDataLoader
from exp_b.evaluator import score_prediction, summarize_scores
from exp_b.model_adapter import build_model
from exp_b.schema import load_config


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
    loader = ReasoningDataLoader()
    rows: list[dict] = []

    for model_spec in config.models:
        model = build_model(model_spec)
        for benchmark in config.benchmarks:
            for task in benchmark.tasks:
                for context_length in config.context_lengths:
                    for case in loader.iter_cases(benchmark, task=task, context_length=context_length):
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
                                "prediction": prediction,
                                "answers": case.answers,
                                "metric": case.metric,
                                "score": score_prediction(prediction, case.answers, case.metric),
                                "metadata": case.metadata,
                            }
                        )

    summary = summarize_scores(rows)
    write_jsonl(config.output_dir / "predictions.jsonl", rows)
    write_json(config.output_dir / "summary.json", summary)
    write_csv(config.output_dir / "by_setting.csv", summary["by_setting"])
    write_csv(config.output_dir / "accuracy_decay_slope.csv", summary["accuracy_decay_slope"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp B reasoning evaluation.")
    parser.add_argument("--config", default="configs/exp_b_reasoning.yaml")
    args = parser.parse_args()
    run(args.config)
