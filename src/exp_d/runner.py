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

from exp_d.client_adapter import SystemSpec, build_client
from exp_d.profiler import summarize_batch
from exp_d.workload import build_workload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows: list[dict] = []
    methods = args.methods if args.model_name.startswith("Llama") else ["baseline"]
    system = SystemSpec(
        name=args.system_name,
        model_name=args.model_name,
        architecture=args.architecture,
        backend=args.backend,
        model_path=args.model_path,
        implementation=args.implementation,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        endpoint=args.endpoint,
    )

    for method in methods:
        client = build_client(system, method=method)
        for context_length in args.context_lengths:
            for batch_size in args.batch_sizes:
                requests = build_workload(
                    context_length=context_length,
                    batch_size=batch_size,
                    output_length=args.output_length,
                )
                responses = client.generate_batch(requests)
                metrics = summarize_batch(responses)
                rows.append(
                    {
                        "system": system.name,
                        "model": system.model_name,
                        "architecture": system.architecture,
                        "backend": system.backend,
                        "method": method,
                        "context_length": context_length,
                        "batch_size": batch_size,
                        "output_length": args.output_length,
                        **metrics,
                    }
                )

    write_csv(output_dir / "serving_metrics.csv", rows)
    write_json(output_dir / "summary.json", {"num_settings": len(rows)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp D serving benchmark.")
    parser.add_argument("--system_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--implementation")
    parser.add_argument("--model_path")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--config_path")
    parser.add_argument("--endpoint")
    parser.add_argument("--methods", nargs="*", default=["baseline"])
    parser.add_argument("--context_lengths", nargs="+", type=int, required=True)
    parser.add_argument("--batch_sizes", nargs="+", type=int, required=True)
    parser.add_argument("--output_length", type=int, default=128)
    parser.add_argument("--output_dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
