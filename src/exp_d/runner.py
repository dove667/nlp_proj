from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from exp_d.client_adapter import build_client
from exp_d.profiler import summarize_batch
from exp_d.schema import load_config
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


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    rows: list[dict] = []

    for system in config.systems:
        methods = config.methods if system.model_name.startswith("Llama") else ["baseline"]
        for method in methods:
            client = build_client(system, method=method)
            for context_length in config.context_lengths:
                for batch_size in config.batch_sizes:
                    requests = build_workload(
                        context_length=context_length,
                        batch_size=batch_size,
                        output_length=config.output_length,
                    )
                    responses = client.generate_batch(requests)
                    metrics = summarize_batch(responses)
                    rows.append(
                        {
                            "experiment": config.experiment,
                            "system": system.name,
                            "model": system.model_name,
                            "architecture": system.architecture,
                            "backend": system.backend,
                            "method": method,
                            "context_length": context_length,
                            "batch_size": batch_size,
                            "output_length": config.output_length,
                            **metrics,
                        }
                    )

    write_csv(config.output_dir / "serving_metrics.csv", rows)
    write_json(config.output_dir / "summary.json", {"num_settings": len(rows), "metrics": config.metrics})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp D serving benchmark.")
    parser.add_argument("--config", default="configs/exp_d_serving.yaml")
    args = parser.parse_args()
    run(args.config)
