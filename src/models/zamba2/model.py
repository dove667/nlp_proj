from __future__ import annotations

from models.base import GenerationOutput
from models.spec import ModelSpec


class Zamba2SourceModel:
    """Placeholder for a local Zamba2 source implementation.

    Zamba2 is a hybrid long-context architecture. The faithful implementation
    should be filled in after confirming the exact checkpoint structure and any
    required custom kernels available on the server.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        raise NotImplementedError("Implement Zamba2 hybrid source model here.")

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
