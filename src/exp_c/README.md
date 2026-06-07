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
  - `continuation prompt + long generation`，`bs=1`
  - 指标：`TPOT`
  - 默认 `prompt_len=256`、`output_len=1024`
  - prompt 会显式要求模型至少生成目标长度，减少提前停止
  - TPOT 按真实生成的新 token 数计算，并额外记录 `avg_actual_output_len`
  - 可选 `--output_lens 256 512 1024` 做小 ablation

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

C2 小 ablation 示例：

```bash
python src/exp_c/bench_decode_hf.py --output_lens 256 512 1024
python src/exp_c/analyze_exp_c.py
```

## 输出文件

- `results/exp_c/prefill_hf.jsonl`
- `results/exp_c/decode_hf.jsonl`
- `results/exp_c/backend_llama.jsonl`

以及分析产物：

- `c1_prefill_ttft_vs_context.pdf`
- `c1_prefill_memory_vs_context.pdf`
- `c2_decode_tpot_compare.pdf`
- `c2_decode_tpot_vs_output_len.pdf`（仅在多 `output_len` 时生成）
- `c3_backend_llama_requests_vs_batch_8k.pdf`
- `c3_backend_llama_requests_vs_batch_16k.pdf`
- `c3_backend_llama_hf_capacity_summary.csv`
