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

**Benchmark**：
- 数据来源：`RULER`
- 当前使用的 RULER NIAH 子任务：`niah_single_1`、`niah_multikey_1`
- 叙事上仍将该组实验称为 **NIAH retrieval**

**长度档位**：4K、8K、16K、32K（Llama 测到 32K；Falcon3-Mamba 在当前环境下稳定测到 16K）

**指标**：accuracy、effective context length $L_\text{eff}$（accuracy ≥ 0.8 的最大长度）

---

## Exp B：跨架构 Reasoning 测试

**问题**：retrieval 能力强的架构，reasoning 能力是否同样强？

**模型**：与 Exp A 完全一致（两个模型）

**Benchmark**：
- 数据来源：`RULER + LongBench`
- `RULER reasoning` 子任务：variable tracking (vt)、common words extraction (cwe)、frequent words extraction (fwe)
- `LongBench` 子集（4 个）：
  - HotpotQA（multi-doc QA，多跳推理）
  - Qasper（单文档学术 QA）
  - GovReport（长文档摘要）
  - RepoBench-P（代码上下文）

**实验定义拆分**：
- `RULER reasoning` 负责测 `reasoning with context scaling`，因此保留 `4K / 8K / 16K / 32K` 长度档位
- `LongBench` 不再做人为长度分档，也不再做 head-tail 截断；它只在原始 benchmark 样本上评测最终任务表现

**指标**：
- `RULER reasoning`：accuracy
- `LongBench`：任务相关指标（QA F1 / Rouge-L F1 / exact match）

**预期发现**：在 RULER retrieval 上表现较强的模型，在合成 reasoning 扩长测试和真实 LongBench 任务上仍可能显著掉点，且不同架构的掉点模式不同。

---

## Exp C：系统性能测试

**问题**：在真实 serving 场景下，长上下文的可用性主要受哪一种系统成本限制：prefill 延迟、decode 成本，还是显存占用？

**模型**：
- Llama-3.1-8B-Instruct（Dense Transformer，主线）
- tiiuae/Falcon3-Mamba-7B-Instruct（SSM，无 KV cache，用 HF 原生代码）

**变量矩阵**：

| 维度 | 取值 |
|---|---|
| 架构 | Llama（Transformer）/ Falcon3-Mamba（SSM） |
| Serving backend（仅 Llama） | HuggingFace Transformers / vLLM |
| Context length | `C1` 为 4K、8K、16K；`C3` 为 8K、16K |
| Batch size | `C1/C2` 固定 1；`C3` 扫 batch |
| Output length | `C1` 用 1 token；`C2` 默认 1024，可做小 ablation；`C3` 用固定长输出 |

**实现脚本**：
1. `bench_prefill_hf.py`：Llama HF vs Mamba HF，测 `TTFT + peak memory`
2. `bench_decode_hf.py`：Llama HF vs Mamba HF，测 `TPOT`
3. `bench_backend_llama.py`：Llama HF vs Llama vLLM，测 `requests/s / completion time / OOM`，并从 HF 子结果汇总 `max batch`

**指标**：
- TTFT（time to first token）
- TPOT（time per output token）
- Peak GPU memory

---

## 模型与方法汇总

| 实验 | 模型 / 方法 |
|---|---|
| A + B | Llama-3.1-8B-Instruct、tiiuae/Falcon3-Mamba-7B-Instruct |
| C | Llama × {HF, vLLM}；Falcon3-Mamba（HF 原生） |

## 当前范围总结

- 数据来源只保留：`RULER + LongBench`
- `RULER` 用于 retrieval 和 synthetic reasoning [https://github.com/NVIDIA/RULER.git](https://github.com/NVIDIA/RULER.git)
- `Exp A` 具体使用 RULER 下的 `niah_single_1` / `niah_multikey_1`
- `Exp B` 具体拆成两部分：RULER reasoning 用于长度扩展下的推理退化测试，LongBench 用于原始真实样本上的长输入理解任务
- `LongBench` 在本项目中不再做人为长度切片，以避免把模型能力和截断策略混在一起
- 不专门纳入“短输入长生成”任务
- 当前正式实验只保留两类经典路线：Transformer（Llama）与 SSM（Falcon3-Mamba）；`Zamba2` 不再纳入评测矩阵
- `Falcon3-Mamba-7B-Instruct` 的 32K 配置不纳入正式结果：在当前 24GB RTX 4090 实验环境中，单卡 OOM，多卡 `device_map=auto` 在 Mamba CUDA kernel 路径上存在稳定性问题
