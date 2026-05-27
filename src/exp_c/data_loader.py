from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtensionCase:
    case_id: str
    benchmark: str
    task: str
    context_length: int
    prompt: str
    answers: list[str]
    metric: str
    metadata: dict


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_path: str
    tasks: list[str]


class ExtensionDataLoader:
    """Server-side adapter reusing prepared RULER and LongBench samples."""

    def iter_cases(
        self,
        benchmark: BenchmarkSpec,
        *,
        task: str,
        context_length: int,
    ) -> Iterable[ExtensionCase]:
        raise NotImplementedError("Implement loading prepared Exp C evaluation cases.")
