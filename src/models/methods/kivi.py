from __future__ import annotations

from models.base import GenerationModel, GenerationOutput
from models.spec import MethodSpec


class KIVIMethod:
    def __init__(self, model: GenerationModel, spec: MethodSpec) -> None:
        self.model = model
        self.spec = spec
        raise NotImplementedError("Implement KIVI asymmetric KV cache quantization here.")

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
