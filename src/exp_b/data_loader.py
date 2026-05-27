from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningCase:
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


class ReasoningDataLoader:
    """Server-side adapter for prepared RULER reasoning and LongBench data."""

    def iter_cases(
        self,
        benchmark: BenchmarkSpec,
        *,
        task: str,
        context_length: int,
    ) -> Iterable[ReasoningCase]:
        raise NotImplementedError(
            "Implement loading prepared RULER reasoning / LongBench cases from benchmark.data_path."
        )
