from __future__ import annotations

from exp_b.schema import ModelSpec
from models.base import GenerationOutput
from models.factory import build_model as build_source_model
from models.spec import ModelSpec as SourceModelSpec


class ReasoningModel:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model = build_source_model(
            SourceModelSpec(
                name=spec.name,
                architecture=spec.architecture,
                implementation=spec.implementation,
                model_path=spec.model_path,
                tokenizer_path=spec.tokenizer_path,
                config_path=spec.config_path,
            )
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        output: GenerationOutput = self.model.generate(prompt, max_new_tokens=max_new_tokens)
        return output.text


def build_model(spec: ModelSpec) -> ReasoningModel:
    return ReasoningModel(spec)
