from __future__ import annotations

from dataclasses import dataclass

from exp_d.schema import SystemSpec
from exp_d.workload import ServingRequest


@dataclass(frozen=True)
class ServingResponse:
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    ttft_seconds: float
    total_seconds: float
    extra: dict


class ServingClient:
    def __init__(self, spec: SystemSpec, *, method: str) -> None:
        self.spec = spec
        self.method = method

    def generate_batch(self, requests: list[ServingRequest]) -> list[ServingResponse]:
        raise NotImplementedError("Connect this to HF / vLLM / SGLang / official model serving.")


def build_client(spec: SystemSpec, *, method: str) -> ServingClient:
    return ServingClient(spec, method=method)
