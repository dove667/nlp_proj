from __future__ import annotations

import time
from dataclasses import dataclass

from exp_c.schema import BaseModelSpec, MethodSpec


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    extra: dict


class InferenceMethod:
    def __init__(self, base_model: BaseModelSpec, method: MethodSpec) -> None:
        self.base_model = base_model
        self.method = method

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationResult:
        start = time.perf_counter()
        raise NotImplementedError(
            "Connect this method to baseline / YaRN / Self-Extend / KIVI / SnapKV / FIER / StreamingLLM."
        )
        latency = time.perf_counter() - start
        return GenerationResult(text="", latency_seconds=latency, extra={})


def build_method(base_model: BaseModelSpec, method: MethodSpec) -> InferenceMethod:
    return InferenceMethod(base_model, method)
