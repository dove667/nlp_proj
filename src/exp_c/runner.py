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

from exp_c.data_loader import BenchmarkSpec, ExtensionDataLoader
from exp_c.evaluator import score_prediction, summarize_methods
from exp_c.method_adapter import BaseModelSpec, MethodSpec, build_method


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


def collect_benchmarks(args: argparse.Namespace) -> list[BenchmarkSpec]:
    benchmarks: list[BenchmarkSpec] = []
    if args.ruler_data_root and args.ruler_niah_tasks:
        benchmarks.append(BenchmarkSpec("ruler_niah", args.ruler_data_root, args.ruler_niah_tasks))
    if args.ruler_data_root and args.ruler_reasoning_tasks:
        benchmarks.append(BenchmarkSpec("ruler_reasoning", args.ruler_data_root, args.ruler_reasoning_tasks))
    if args.longbench_data_root and args.longbench_tasks:
        benchmarks.append(BenchmarkSpec("longbench", args.longbench_data_root, args.longbench_tasks))
    if not benchmarks:
        raise ValueError("Provide at least one benchmark via --ruler_data_root or --longbench_data_root.")
    return benchmarks


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    loader = ExtensionDataLoader()
    base_model = BaseModelSpec(
        name=args.model_name,
        architecture=args.architecture,
        implementation=args.implementation,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
    )
    method = build_method(
        base_model,
        MethodSpec(
            name=args.method_name,
            implementation=args.method_implementation,
            config_path=args.method_config_path,
        ),
    )
    rows: list[dict] = []

    for benchmark in collect_benchmarks(args):
        for task in benchmark.tasks:
            for context_length in args.context_lengths:
                for case in loader.iter_cases(benchmark, task=task, context_length=context_length):
                    result = method.generate(case.prompt, max_new_tokens=args.max_new_tokens)
                    rows.append(
                        {
                            "base_model": base_model.name,
                            "method": args.method_name,
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
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "by_setting.csv", summary["by_setting"])
    write_csv(output_dir / "method_gains.csv", summary["method_gains"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp C inference-time extension evaluation.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--config_path")
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--method_implementation", required=True)
    parser.add_argument("--method_config_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_lengths", nargs="+", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--ruler_data_root")
    parser.add_argument("--ruler_niah_tasks", nargs="*")
    parser.add_argument("--ruler_reasoning_tasks", nargs="*")
    parser.add_argument("--longbench_data_root")
    parser.add_argument("--longbench_tasks", nargs="*")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
