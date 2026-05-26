from __future__ import annotations

import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from models.base import GenerationOutput
from models.llama.configuration import LlamaConfig
from models.spec import ModelSpec


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x), self.num_heads)
        k = self._shape(self.k_proj(x), self.num_kv_heads)
        v = self._shape(self.v_proj(x), self.num_kv_heads)
        cos, sin = self.rope(seq_len, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.lm_head(self.norm(x))


class LlamaSourceModel:
    """Local Llama source implementation.

    This is intentionally source code in this repo, not a HuggingFace AutoModel wrapper.
    Weight loading and tokenizer binding are still format-dependent and should be
    completed once the server weight layout is fixed.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        config_path = spec.config_path or str(Path(spec.model_path) / "config.json")
        self.config = LlamaConfig.from_json(config_path)
        self.model = LlamaForCausalLM(self.config)
        self.tokenizer = None
        self._load_weights(Path(spec.model_path))
        self.model.eval()

    def _load_weights(self, model_path: Path) -> None:
        raise NotImplementedError(
            "Map local Llama-3.1 checkpoint tensors into LlamaForCausalLM.state_dict(). "
            "Do this after confirming whether weights are .pth, .bin, or .safetensors."
        )

    def _encode(self, prompt: str) -> torch.Tensor:
        raise NotImplementedError("Bind the local Llama-3 tokenizer here; do not use AutoTokenizer.")

    def _decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError("Bind the local Llama-3 tokenizer here; do not use AutoTokenizer.")

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> GenerationOutput:
        start = time.perf_counter()
        input_ids = self._encode(prompt)
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.model(generated)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if int(next_token.item()) == self.config.eos_token_id:
                break
        new_tokens = generated[0, input_ids.shape[-1] :].tolist()
        return GenerationOutput(
            text=self._decode(new_tokens),
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=len(new_tokens),
            latency_seconds=time.perf_counter() - start,
            extra={},
        )
