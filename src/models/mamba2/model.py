from __future__ import annotations

from models.base import GenerationOutput
from models.spec import ModelSpec


class Mamba2SourceModel:
    """Placeholder for a local Mamba-2 source implementation.

    Mamba-2 depends on selective scan kernels and checkpoint-specific parameter
    naming. This class is intentionally left for the server implementation rather
    than pretending a small pure-Python version would be faithful.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        raise NotImplementedError("Implement Mamba-2 blocks and selective scan kernel binding here.")

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        raise NotImplementedError
