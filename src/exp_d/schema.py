from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SystemSpec:
    name: str
    model_name: str
    architecture: str
    backend: str
    model_path: str | None = None
    endpoint: str | None = None


@dataclass(frozen=True)
class ExpDConfig:
    experiment: str
    description: str
    output_dir: Path
    systems: list[SystemSpec]
    methods: list[str]
    context_lengths: list[int]
    batch_sizes: list[int]
    output_length: int
    metrics: list[str] = field(default_factory=list)


def load_config(path: str | Path) -> ExpDConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ExpDConfig(
        experiment=raw["experiment"],
        description=raw.get("description", ""),
        output_dir=Path(raw["output_dir"]),
        systems=[SystemSpec(**item) for item in raw["systems"]],
        methods=list(raw.get("methods", ["baseline"])),
        context_lengths=list(raw["context_lengths"]),
        batch_sizes=list(raw["batch_sizes"]),
        output_length=int(raw["output_length"]),
        metrics=list(raw.get("metrics", [])),
    )
