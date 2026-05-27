from __future__ import annotations

from dataclasses import dataclass

from models.factory import build_method as build_source_method
from models.factory import build_model
from models.spec import MethodSpec as SourceMethodSpec
from models.spec import ModelSpec as SourceModelSpec


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    extra: dict


@dataclass(frozen=True)
class BaseModelSpec:
    name: str
    model_path: str
    architecture: str
    implementation: str
    tokenizer_path: str | None = None
    config_path: str | None = None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    implementation: str
    config_path: str | None = None
    init_kwargs: dict | None = None


class InferenceMethod:
    def __init__(self, base_model: BaseModelSpec, method: MethodSpec) -> None:
        self.base_model = base_model
        self.method = method
        source_model = build_model(
            SourceModelSpec(
                name=base_model.name,
                architecture=base_model.architecture,
                implementation=base_model.implementation,
                model_path=base_model.model_path,
                tokenizer_path=base_model.tokenizer_path,
                config_path=base_model.config_path,
            )
        )
        self.impl = build_source_method(
            source_model,
            SourceMethodSpec(
                name=method.name,
                implementation=method.implementation,
                config_path=method.config_path,
                init_kwargs=method.init_kwargs,
            ),
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationResult:
        output = self.impl.generate(prompt, max_new_tokens=max_new_tokens)
        return GenerationResult(
            text=output.text,
            latency_seconds=float(output.latency_seconds or 0.0),
            extra=output.extra or {},
        )


def build_method(base_model: BaseModelSpec, method: MethodSpec) -> InferenceMethod:
    return InferenceMethod(base_model, method)
