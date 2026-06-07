# Exp C — 按三个脚本收紧

`exp_c` 收紧为 3 个 benchmark 脚本和 1 个总分析脚本。

## Benchmark 脚本

- `bench_prefill_hf.py`
  - C1 主实验
  - `Llama HF vs Mamba HF`
  - 单卡，`4K/8K/16K`，`bs=1`
  - 指标：`TTFT`、`peak memory`

- `bench_decode_hf.py`
  - C2 decode 特性
  - `Llama HF vs Mamba HF`
  - `continuation prompt + fixed-length generation`，`bs=1`
  - 指标：`TPOT`
  - 固定 `output_len=128`，扫 `prompt_len`
  - 默认 `256/4K/8K/16K`
  - prompt 会显式要求模型至少生成目标长度，减少提前停止
  - TPOT 按真实生成的新 token 数计算，并额外记录 `avg_actual_output_len`

- `bench_backend_llama.py`
  - C3 服务可用性/调度对比
  - `Llama HF vs Llama vLLM`
  - 固定 `8K/16K`，扫 `batch`
  - 指标：`requests/s`、`avg completion time`、`OOM`
  - HF 的 `max supported batch` 由这份结果直接汇总，不再单独跑一份脚本

## Analysis 脚本

- `analyze_exp_c.py`
  - 统一读取以上 3 份 JSONL
  - 输出全部图和 summary CSV

## 运行方式

```bash
python src/exp_c/bench_prefill_hf.py
python src/exp_c/bench_decode_hf.py
python src/exp_c/bench_backend_llama.py
python src/exp_c/analyze_exp_c.py
```

C2 prompt-length sweep 示例：

```bash
python src/exp_c/bench_decode_hf.py --prompt_lens 256 4096 8192 16384
python src/exp_c/analyze_exp_c.py
```

## 输出文件

- `results/exp_c/prefill_hf.jsonl`
- `results/exp_c/decode_hf.jsonl`
- `results/exp_c/backend_llama.jsonl`

以及分析产物：

- `c1_prefill_ttft_vs_context.pdf`
- `c1_prefill_memory_vs_context.pdf`
- `c2_decode_tpot_vs_prompt_len.pdf`
- `c3_backend_llama_requests_vs_batch_8k.pdf`
- `c3_backend_llama_requests_vs_batch_16k.pdf`
- `c3_backend_llama_hf_capacity_summary.csv`
