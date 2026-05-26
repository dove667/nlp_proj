from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from exp_a.schema import BenchmarkSpec


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    benchmark: str
    task: str
    context_length: int
    needle_position: float | None
    prompt: str
    answer: str
    metadata: dict


class RetrievalDataLoader:
    """Server-side adapter for prepared NIAH and RULER retrieval data."""

    def iter_cases(
        self,
        benchmark: BenchmarkSpec,
        *,
        task: str,
        context_length: int,
        needle_positions: list[float],
    ) -> Iterable[RetrievalCase]:
        raise NotImplementedError(
            "Implement loading prepared NIAH/RULER retrieval cases from benchmark.data_path."
        )
