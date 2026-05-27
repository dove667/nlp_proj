from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exp_b.data_loader import BenchmarkSpec, ReasoningDataLoader
from exp_b.evaluator import score_prediction, summarize_scores
from exp_b.model_adapter import ModelSpec, build_model


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


def add_benchmark_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    benchmarks: list[BenchmarkSpec] = []
    if args.ruler_data_root and args.ruler_tasks:
        benchmarks.append(
            BenchmarkSpec(
                name="ruler_reasoning",
                data_path=args.ruler_data_root,
                tasks=args.ruler_tasks,
            )
        )
    if args.longbench_data_root and args.longbench_tasks:
        benchmarks.append(
            BenchmarkSpec(
                name="longbench",
                data_path=args.longbench_data_root,
                tasks=args.longbench_tasks,
            )
        )
    if not benchmarks:
        raise ValueError("Provide at least one benchmark via --ruler_data_root/--ruler_tasks or --longbench_data_root/--longbench_tasks.")
    return benchmarks


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    loader = ReasoningDataLoader()
    model_spec = ModelSpec(
        name=args.model_name,
        architecture=args.architecture,
        implementation=args.implementation,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
    )
    model = build_model(model_spec)
    rows: list[dict] = []

    for benchmark in add_benchmark_specs(args):
        for task in benchmark.tasks:
            for context_length in args.context_lengths:
                for case in loader.iter_cases(benchmark, task=task, context_length=context_length):
                    prediction = model.generate(case.prompt, max_new_tokens=args.max_new_tokens)
                    rows.append(
                        {
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
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "by_setting.csv", summary["by_setting"])
    write_csv(output_dir / "accuracy_decay_slope.csv", summary["accuracy_decay_slope"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp B reasoning evaluation.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--config_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_lengths", nargs="+", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--ruler_data_root")
    parser.add_argument("--ruler_tasks", nargs="*")
    parser.add_argument("--longbench_data_root")
    parser.add_argument("--longbench_tasks", nargs="*")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
