from __future__ import annotations

from exp_b.schema import ModelSpec


class ReasoningModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        raise NotImplementedError("Connect this to server-side model inference.")


def build_model(spec: ModelSpec) -> ReasoningModel:
    return ReasoningModel(spec)
