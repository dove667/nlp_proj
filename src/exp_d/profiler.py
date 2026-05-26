from __future__ import annotations

from exp_d.client_adapter import ServingResponse


def summarize_batch(responses: list[ServingResponse], *, peak_gpu_memory_gb: float | None = None) -> dict:
    if not responses:
        return {
            "ttft": 0.0,
            "tpot": 0.0,
            "throughput": 0.0,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }

    total_output_tokens = sum(response.output_tokens for response in responses)
    max_total_seconds = max(response.total_seconds for response in responses)
    mean_ttft = sum(response.ttft_seconds for response in responses) / len(responses)
    decode_seconds = sum(
        max(0.0, response.total_seconds - response.ttft_seconds) for response in responses
    )
    tpot = decode_seconds / max(1, total_output_tokens)
    throughput = total_output_tokens / max(max_total_seconds, 1e-9)
    return {
        "ttft": mean_ttft,
        "tpot": tpot,
        "throughput": throughput,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }
