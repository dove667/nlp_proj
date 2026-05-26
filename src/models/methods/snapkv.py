from __future__ import annotations

from models.base import GenerationModel, GenerationOutput
from models.spec import MethodSpec


class SnapKVMethod:
    def __init__(self, model: GenerationModel, spec: MethodSpec) -> None:
        self.model = model
        self.spec = spec
        raise NotImplementedError("Implement SnapKV prefill-time KV selection here.")

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
