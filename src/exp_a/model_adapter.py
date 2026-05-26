from __future__ import annotations

from exp_a.schema import ModelSpec


class RetrievalModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def generate(self, prompt: str, *, max_new_tokens: int = 64) -> str:
        raise NotImplementedError("Connect this to HF / official model inference on the server.")


def build_model(spec: ModelSpec) -> RetrievalModel:
    return RetrievalModel(spec)
