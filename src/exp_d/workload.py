from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    prompt: str
    context_length: int
    output_length: int
    metadata: dict


def build_workload(*, context_length: int, batch_size: int, output_length: int) -> list[ServingRequest]:
    filler = "Long context serving benchmark sentence."
    prompt = " ".join([filler] * max(1, context_length // len(filler.split())))
    return [
        ServingRequest(
            request_id=f"ctx{context_length}-bs{batch_size}-{idx}",
            prompt=prompt,
            context_length=context_length,
            output_length=output_length,
            metadata={},
        )
        for idx in range(batch_size)
    ]
