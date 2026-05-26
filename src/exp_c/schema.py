from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BaseModelSpec:
    name: str
    model_path: str
    architecture: str
    implementation: str
    tokenizer_path: str | None = None
    config_path: str | None = None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    implementation: str
    config_path: str | None = None
    init_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_path: str
    tasks: list[str]


@dataclass(frozen=True)
class ExpCConfig:
    experiment: str
    description: str
    output_dir: Path
    base_model: BaseModelSpec
    methods: list[MethodSpec]
    benchmarks: list[BenchmarkSpec]
    context_lengths: list[int]
    metrics: list[str] = field(default_factory=list)


def load_config(path: str | Path) -> ExpCConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ExpCConfig(
        experiment=raw["experiment"],
        description=raw.get("description", ""),
        output_dir=Path(raw["output_dir"]),
        base_model=BaseModelSpec(**raw["base_model"]),
        methods=[MethodSpec(**item) for item in raw["methods"]],
        benchmarks=[BenchmarkSpec(**item) for item in raw["benchmarks"]],
        context_lengths=list(raw["context_lengths"]),
        metrics=list(raw.get("metrics", [])),
    )
