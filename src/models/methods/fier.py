from __future__ import annotations

from models.base import GenerationModel, GenerationOutput
from models.spec import MethodSpec


class FIERMethod:
    def __init__(self, model: GenerationModel, spec: MethodSpec) -> None:
        self.model = model
        self.spec = spec
        raise NotImplementedError(
            "FIER source code is not confirmed. Implement this method after choosing a concrete implementation."
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
