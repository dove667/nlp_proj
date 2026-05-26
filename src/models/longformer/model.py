from __future__ import annotations

from models.base import GenerationOutput
from models.spec import ModelSpec


class LongformerSourceModel:
    """Placeholder for a local Longformer source implementation.

    Longformer is encoder-style and not a normal causal chat model. For Exp A/B it
    likely needs task-specific heads or a prompt-to-classification wrapper. Leave
    the exact implementation until the prepared task format is fixed.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        raise NotImplementedError("Implement local Longformer attention/model code or task head here.")

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
