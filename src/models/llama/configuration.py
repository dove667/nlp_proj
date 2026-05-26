from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    @classmethod
    def from_json(cls, path: str | Path) -> "LlamaConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in allowed})
