from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    implementation: str
    model_path: str
    tokenizer_path: str | None = None
    config_path: str | None = None
    init_kwargs: dict[str, Any] | None = None
    generation_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    implementation: str
    config_path: str | None = None
    init_kwargs: dict[str, Any] | None = None
