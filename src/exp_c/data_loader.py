from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from exp_c.schema import BenchmarkSpec


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


class ExtensionDataLoader:
    """Server-side adapter reusing prepared NIAH, RULER, and LongBench samples."""

    def iter_cases(
        self,
        benchmark: BenchmarkSpec,
        *,
        task: str,
        context_length: int,
    ) -> Iterable[ExtensionCase]:
        raise NotImplementedError("Implement loading prepared Exp C evaluation cases.")
