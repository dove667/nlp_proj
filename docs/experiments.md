# 实验规划

## 核心问题

长上下文能力由四个维度共同决定：名义窗口长度、有效检索长度、长程推理能力、系统效率。四组实验分别拆解这四个维度。

## 数据来源约定

为减少数据准备和评测口径分散，当前实验统一只使用两套 benchmark：

- `RULER`：负责 synthetic retrieval / reasoning 测试
- `LongBench`：负责更真实的长文档理解、问答、摘要与代码上下文测试

当前版本**不再单独维护独立 NIAH 数据集**。不过 `Exp A` 仍然会明确做 **NIAH retrieval 任务**，只是这些样本来自 **RULER 目录下的 NIAH 子任务**。另外，本项目的关注重点是“长输入理解”，因此不专门设计“短输入长生成”实验。

---

## Exp A：跨架构 Retrieval 基线

**问题**：不同序列建模架构在 retrieval 任务上的有效上下文长度有何差异？研究问题不是"谁能接受更长的输入"，而是"在各自支持的长度范围内，哪种架构能更可靠地检索到目标信息"。参照 RULER 原论文，Llama-3.1-8B 声称支持 128K 但有效上下文约为 32K，因此以 32K 为实验上限。

**模型**（每类选最典型的一个）：

| 架构类型 | 模型 | 官方支持长度 |
|---|---|---|
| Dense Transformer | Meta/Llama-3.1-8B-Instruct | 128K（有效约 32K） |
| SSM / linear recurrent | tiiuae/Falcon3-Mamba-7B-Instruct | 32K |
| Hybrid (Mamba + Transformer) | Zyphra/Zamba2-7B-Instruct-v2 | 16K（可扩展） |

**Benchmark**：
- 数据来源：`RULER`
- 当前使用的 RULER NIAH 子任务：`niah_single_1`、`niah_multikey_1`
- 叙事上仍将该组实验称为 **NIAH retrieval**

**长度档位**：4K、8K、16K、32K（三个模型统一测到 32K，Zamba2 超出官方支持范围的部分正好展示 degradation）

**指标**：accuracy、effective context length $L_\text{eff}$（accuracy ≥ 0.8 的最大长度）

---

## Exp B：跨架构 Reasoning 测试

**问题**：retrieval 能力强的架构，reasoning 能力是否同样强？

**模型**：与 Exp A 完全一致（三个模型）

**Benchmark**：
- 数据来源：`RULER + LongBench`
- RULER reasoning 子任务：variable tracking、common words aggregation、multi-hop tracing
- LongBench 子集（4 个）：
  - HotpotQA（multi-doc QA，多跳推理）
  - Qasper（单文档学术 QA）
  - GovReport（长文档摘要）
  - RepoBench-P（代码上下文）

**长度档位**：8K、16K、32K（三个模型统一测到 32K，Zamba2 超出官方支持范围的部分正好展示 degradation）

**指标**：accuracy / F1（任务相关）、accuracy decay slope $\Delta(L_1, L_2)$

**预期发现**：在 RULER retrieval 上表现较强的模型，在 RULER aggregation / multi-hop tracing 和 LongBench 多文档理解任务上仍可能显著掉点，且不同架构的掉点模式不同。

---

## Exp C：推理时优化方法测试

**问题**：不改变模型权重的推理时方法能在多大程度上提升 Transformer 的有效上下文长度？对 retrieval 和 reasoning 的提升是否对称？

**基础模型**：Llama-3.1-8B-Instruct（仅 Transformer，因为这些方法都针对 Transformer KV cache 设计）

**方法**（每类选一个）：

| 类别 | 方法 | 改变什么 |
|---|---|---|
| Baseline | 原生推理 | — |
| Positional extension | YaRN | 可接受长度（RoPE 插值） |
| Attention reorganization | Self-Extend | 注意力结构（bi-level attention） |
| KV quantization | KIVI | KV cache 数值精度（2-bit 非对称量化） |
| KV eviction | SnapKV | prefill 阶段 KV 选择 |
| KV retrieval | FIER | 每步 token 级 KV 检索 |
| Streaming | StreamingLLM | sink + sliding window |

**Benchmark**：
- 复用 Exp A 的 **RULER NIAH retrieval 子任务**：`niah_single_1`、`niah_multikey_1`
- 复用 Exp B 的 **RULER reasoning 子任务**：`common words aggregation`、`multi-hop tracing`
- 复用 Exp B 的 LongBench 子集：优先 `HotpotQA`，必要时补 `Qasper`

**长度档位**：32K、64K、128K（超出 Llama 原生舒适区的档位）

**横向参考**：在同一张图中保留 Exp A/B 中 Falcon3-Mamba 的结果作为参考线，直观对比"架构效率"与"推理时优化"两条路线的收益

**指标**：accuracy gain vs. baseline、$L_\text{eff}$ 变化、latency overhead

---

## Exp D：系统性能测试

**问题**：在真实 serving 场景下，不同架构、不同推理后端、不同推理优化方法的吞吐与延迟如何？

**模型**：
- Llama-3.1-8B-Instruct（Dense Transformer，主线）
- tiiuae/Falcon3-Mamba-7B-Instruct（SSM，无 KV cache，用 HF 原生代码）
- Zyphra/Zamba2-7B-Instruct-v2（Hybrid，用 HF 原生代码）

**变量矩阵**：

| 维度 | 取值 |
|---|---|
| 架构 | Llama（Transformer）/ Falcon3-Mamba（SSM）/ Zamba2（Hybrid） |
| Serving backend（仅 Llama） | HuggingFace Transformers / vLLM / SGLang |
| 推理优化方法（仅 Llama） | baseline / KIVI / SnapKV / FIER / StreamingLLM |
| Context length | 4K、8K、16K、32K |
| Batch size | 1、4、8、16 |
| Output length | 固定 128 tokens |

**指标**：
- TTFT（time to first token）
- TPOT（time per output token）
- Throughput（tokens/s）
- Peak GPU memory

**核心图**：
1. TTFT vs. context length（Llama 不同 backend + Falcon3-Mamba + Zamba2 并列）
2. Peak memory vs. context length（同上，最能体现 SSM 无 KV cache 增长的优势）
3. Throughput vs. batch size（Llama 不同 backend）
4. Peak memory vs. context length（Llama × 不同推理优化方法）
5. Accuracy vs. latency trade-off（Exp C 的 accuracy + Exp D 的 latency，每个方法一个点）

---

## 模型与方法汇总

| 实验 | 模型 / 方法 |
|---|---|
| A + B | Llama-3.1-8B-Instruct、tiiuae/Falcon3-Mamba-7B-Instruct、Zyphra/Zamba2-7B-Instruct-v2 |
| C | Llama-3.1-8B-Instruct × {baseline, YaRN, Self-Extend, KIVI, SnapKV, FIER, StreamingLLM} |
| D | Llama × {HF, vLLM, SGLang} × {baseline, KIVI, SnapKV, FIER, StreamingLLM}；Falcon3-Mamba、Zamba2（HF 原生） |

## 当前范围总结

- 数据来源只保留：`RULER + LongBench`
- `RULER` 用于 retrieval 和 synthetic reasoning [https://github.com/NVIDIA/RULER.git](https://github.com/NVIDIA/RULER.git)
- `Exp A` 具体使用 RULER 下的 `niah_single_1` / `niah_multikey_1`
- `Exp B` 具体使用 RULER 的 reasoning 子任务与 LongBench 子集
- `LongBench` 用于更真实的长输入理解任务
- 不专门纳入“短输入长生成”任务
