from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    implementation: str
    model_path: str
    tokenizer_path: str | None = None
    config_path: str | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_path: str
    tasks: list[str]


@dataclass(frozen=True)
class ExpBConfig:
    experiment: str
    description: str
    output_dir: Path
    models: list[ModelSpec]
    benchmarks: list[BenchmarkSpec]
    context_lengths: list[int]
    metrics: list[str] = field(default_factory=list)


def load_config(path: str | Path) -> ExpBConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ExpBConfig(
        experiment=raw["experiment"],
        description=raw.get("description", ""),
        output_dir=Path(raw["output_dir"]),
        models=[ModelSpec(**item) for item in raw["models"]],
        benchmarks=[BenchmarkSpec(**item) for item in raw["benchmarks"]],
        context_lengths=list(raw["context_lengths"]),
        metrics=list(raw.get("metrics", [])),
    )
