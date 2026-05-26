from __future__ import annotations

from models.base import GenerationModel, GenerationOutput
from models.spec import MethodSpec


class BaselineMethod:
    def __init__(self, model: GenerationModel, spec: MethodSpec | None = None) -> None:
        self.model = model
        self.spec = spec

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        return self.model.generate(prompt, max_new_tokens=max_new_tokens)
