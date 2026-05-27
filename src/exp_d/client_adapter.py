from __future__ import annotations

from dataclasses import dataclass

from models.factory import build_model
from models.spec import ModelSpec
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


@dataclass(frozen=True)
class SystemSpec:
    name: str
    model_name: str
    architecture: str
    backend: str
    model_path: str | None = None
    implementation: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None
    endpoint: str | None = None


class ServingClient:
    def __init__(self, spec: SystemSpec, *, method: str) -> None:
        self.spec = spec
        self.method = method
        self.model = None
        if spec.backend == "source":
            if not all([spec.model_path, spec.implementation]):
                raise ValueError(f"Source backend is missing source fields for {spec.name}.")
            self.model = build_model(
                ModelSpec(
                    name=spec.model_name,
                    architecture=spec.architecture,
                    implementation=str(spec.implementation),
                    model_path=str(spec.model_path),
                    tokenizer_path=spec.tokenizer_path,
                    config_path=spec.config_path,
                )
            )

    def generate_batch(self, requests: list[ServingRequest]) -> list[ServingResponse]:
        if self.spec.backend == "source":
            assert self.model is not None
            responses = []
            for request in requests:
                output = self.model.generate(request.prompt, max_new_tokens=request.output_length)
                responses.append(
                    ServingResponse(
                        request_id=request.request_id,
                        text=output.text,
                        input_tokens=int(output.input_tokens or request.context_length),
                        output_tokens=int(output.output_tokens or request.output_length),
                        ttft_seconds=float(output.extra.get("ttft_seconds", 0.0))
                        if output.extra
                        else 0.0,
                        total_seconds=float(output.latency_seconds or 0.0),
                        extra=output.extra or {},
                    )
                )
            return responses

        if self.spec.backend in {"vllm", "sglang"}:
            raise NotImplementedError(
                f"Implement {self.spec.backend} HTTP/OpenAI-compatible client for endpoint {self.spec.endpoint}."
            )

        raise ValueError(f"Unsupported serving backend: {self.spec.backend}")


def build_client(spec: SystemSpec, *, method: str) -> ServingClient:
    return ServingClient(spec, method=method)
