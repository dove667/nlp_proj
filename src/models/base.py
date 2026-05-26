from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float | None = None
    extra: dict | None = None


class GenerationModel(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        """Generate text from one prompt."""


class InferenceMethod(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        """Generate text with a specific inference-time method."""
