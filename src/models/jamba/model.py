from __future__ import annotations

from models.base import GenerationOutput
from models.spec import ModelSpec


class JambaSourceModel:
    """Placeholder for a local Jamba source implementation.

    Jamba combines attention, Mamba-like blocks, and MoE routing. The faithful
    implementation should be filled in after confirming the exact checkpoint and
    routing format available on the server.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        raise NotImplementedError("Implement Jamba hybrid/MoE source model here.")

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
