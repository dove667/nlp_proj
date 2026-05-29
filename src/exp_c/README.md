# Exp C — 系统性能基准

衡量不同后端与架构下的 TTFT、TPOT、吞吐量与 GPU 峰值显存。

## 变量矩阵

| 维度 | 取值 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct, Falcon3-Mamba-7B-Instruct |
| 后端 | HF Transformers（两种模型）, vLLM（仅 Llama） |
| 上下文长度 | 4K, 8K, 16K, 32K |
| 批大小 | 1, 4, 8, 16 |
| 输出长度 | 128 tokens（固定） |

## 脚本

| 脚本 | 作用 |
|---|---|
| `bench_hf.py` | HF Transformers 后端 — Llama 与 Mamba |
| `bench_vllm.py` | vLLM 后端 — 仅 Llama |
| `analyze_exp_c.py` | 读取结果并生成 3 张核心图表 |

## 运行方式

```bash
conda activate nlp_proj

python src/exp_c/bench_hf.py --model mamba --context_lens 4096 8192 16384
python src/exp_c/bench_hf.py --model llama31
python src/exp_c/bench_vllm.py
python src/exp_c/analyze_exp_c.py
```

结果保存在 `results/exp_c/{model}_{backend}.jsonl`。
图表以 PDF 矢量图输出到 `results/exp_c/`。

## 图表与结果分析

**图 1 — TTFT vs 上下文长度** (`fig1_ttft_vs_context.pdf`, batch size = 1)

观察：Mamba (HF) 的 TTFT 在所有长度档位上均高于 Llama (HF)，三者中最慢（4K=572ms vs Llama HF 437ms）。Llama vLLM 最快（4K=379ms）。

这与"SSM prefill 是 O(n)、Transformer 是 O(n²)"的理论预期相反。可能的解释：Falcon3-Mamba 有 64 层，Llama-3.1-8B 只有 32 层，逐层开销叠加后抵消了复杂度优势；此外 Transformer 的 prefill 大量依赖 cuBLAS 矩阵乘法，硬件优化极为成熟，而 Mamba 的 selective scan kernel 在当前 GPU 上的实际吞吐尚不及前者，导致理论复杂度优势未能转化为实际延迟优势。待进一步验证。

**图 2 — 峰值显存 vs 上下文长度** (`fig2_memory_vs_context.pdf`, batch size = 1)

观察：Mamba (HF) 的峰值显存随上下文增长比 Llama (HF) 更快（4K=16.4GB → 16K=21.8GB，增量 5.4GB；Llama HF 同区间增量 3.1GB）。Llama vLLM 几乎恒定在 21.8GB。

Mamba 增长更快同样与"无 KV cache"的预期相反。可能的解释：KV cache 的优势只在 decode 阶段成立（decode 时 SSM state 大小固定，不随序列增长）；prefill 阶段仍需对整个序列做前向传播，中间激活值随序列长度和层数线性增长，64 层的 Mamba 激活内存反而更大。待进一步验证。

Llama vLLM 恒定是因为启动时按 `gpu_memory_utilization=0.90` 一次性预分配了整个 KV cache pool，`torch.cuda.max_memory_allocated()` 测到的是这个固定上限，与实际使用量无关。

**图 3 — 吞吐量 vs 批大小** (`fig3_throughput_vs_batch.pdf`, ctx = 8K)

观察：Llama HF 在 bs=4 时已 OOM，图上只有 bs=1 一个数据点（~25 tok/s）。Llama vLLM 在 bs=1 到 bs=8 之间吞吐从 40 升至 87 tok/s，bs=8 之后趋于平稳。

注意：两者的对比不完全在同一维度上。HF 是真正的并行 batch，所有请求的 KV cache 同时分配；vLLM 在 KV cache pool 耗尽后会将请求排队串行处理（可从 32K 数据验证：bs=1/4/8/16 的 throughput 几乎相同，约 17.9 tok/s，说明实际是串行执行的）。因此 vLLM 的"不 OOM"是以排队延迟为代价换来的，而非单纯的内存效率提升。这个对比展示的是 serving 场景下两种策略的取舍：HF 要么全部并行要么 OOM，vLLM 通过调度保证了可用性。

## 备注

- 全部实验均为单卡（cuda:0）— Mamba 在 vLLM 中不支持 TP/PP；两种模型都可放入 24 GB
- Mamba 32K 被排除（单卡 4090 OOM，与 Exp A/B 一致）
- vLLM 版本：0.7.3（兼容 torch 2.5.1+cu121 / CUDA 12.1）
