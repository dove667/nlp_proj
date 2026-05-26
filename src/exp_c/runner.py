from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from exp_c.data_loader import ExtensionDataLoader
from exp_c.evaluator import score_prediction, summarize_methods
from exp_c.method_adapter import build_method
from exp_c.schema import load_config


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
    loader = ExtensionDataLoader()
    rows: list[dict] = []

    for method_spec in config.methods:
        method = build_method(config.base_model, method_spec)
        for benchmark in config.benchmarks:
            for task in benchmark.tasks:
                for context_length in config.context_lengths:
                    for case in loader.iter_cases(benchmark, task=task, context_length=context_length):
                        result = method.generate(case.prompt)
                        rows.append(
                            {
                                "experiment": config.experiment,
                                "base_model": config.base_model.name,
                                "method": method_spec.name,
                                "benchmark": benchmark.name,
                                "task": task,
                                "case_id": case.case_id,
                                "context_length": case.context_length,
                                "prediction": result.text,
                                "answers": case.answers,
                                "metric": case.metric,
                                "score": score_prediction(result.text, case.answers, case.metric),
                                "latency_seconds": result.latency_seconds,
                                "extra": result.extra,
                                "metadata": case.metadata,
                            }
                        )

    summary = summarize_methods(rows)
    write_jsonl(config.output_dir / "predictions.jsonl", rows)
    write_json(config.output_dir / "summary.json", summary)
    write_csv(config.output_dir / "by_setting.csv", summary["by_setting"])
    write_csv(config.output_dir / "method_gains.csv", summary["method_gains"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp C inference-time extension evaluation.")
    parser.add_argument("--config", default="configs/exp_c_inference_extension.yaml")
    args = parser.parse_args()
    run(args.config)
