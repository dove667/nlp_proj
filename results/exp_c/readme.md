# Exp C 调查报告：为什么当前结果里 Mamba 没有比 Llama 更省、更快

## 1. 结论先行

这组 `exp_c` 结果整体上是“可以解释的”，但**不能直接按论文里的渐近复杂度叙事去读**。当前结果混合了三层因素：

1. **模型本身的设计差异**
   - `Llama-3.1-8B-Instruct`：`32` 层，`hidden_size=4096`，`num_key_value_heads=8`
   - `Falcon3-Mamba-7B-Instruct`：`64` 层，`hidden_size=4096`，`intermediate_size=8192`，`state_size=16`，`conv_kernel=4`
   - Mamba 这类模型虽然没有 Transformer 式的 KV cache，但它不是“同样深度下更轻”的替代物；当前这个 Falcon Mamba 版本本身就更深，常数项不小。

2. **Hugging Face 当前实现路径的差异**
   - 你现在的 `C1` 不是在测“裸 prefill forward”，而是在测 `model.generate(max_new_tokens=1)`。
   - 在 `transformers==4.49.0` 里，`LlamaForCausalLM.forward` 支持 `logits_to_keep`，而 `GenerationMixin` 会在 generation 时自动传 `logits_to_keep=1`。
   - 但 `FalconMambaForCausalLM.forward` **不支持** `logits_to_keep`，并且它会把整段序列的 logits 做出来，还显式 `.float()` 成 `fp32`。
   - 这个实现差异会**系统性地抬高 Mamba 的 prefill memory 和 TTFT**，而且这不是 Mamba 理论本身的问题，而是当前 HF 路径的问题。

3. **system / kernel 层因素**
   - Chen 机器上 `nlp` 环境是 `torch 2.5.1 + cu121`，`transformers 4.49.0`，`mamba_ssm 2.2.4`，`causal_conv1d 1.6.2.post1`，`triton 3.1.0`。
   - `vllm` 环境是 `vllm 0.7.3`，也明显落后于当前最新版。
   - 当前 Mamba 路径虽然启用了 fast path，但 `mamba_ssm` / `transformers` 这条链路的实现成熟度，明显不如 Llama 走的 PyTorch SDPA 路径。

**因此，当前数据最合理的解读是：**

- `C1` 里 Mamba prefill memory 随上下文增长变快，并不反驳“Mamba 的 recurrent state 是固定大小”这件事；它增长的主要不是 recurrent state，而是**随序列长度增长的中间激活、scan 工作区、以及整段 logits**。
- `C1` 里两个模型 TTFT 都近似线性，也不意味着“Transformer prefill 真变成线性了”；更准确的说法是：**你当前测到的是 HF generate 端到端路径**，其中大量线性项和 fused kernel 行为把理论上的平方注意力复杂度盖住了。
- `C2` 里 Mamba decode TPOT 更慢，是因为当前实验的 `prompt_len≈240` 太短，没有打到 Mamba 的长上下文 decode 优势区间；与此同时，Mamba 的**每 token 常数项**在这套实现上又明显更大。

---

## 2. 原始实验到底在测什么

### 2.1 `C1` 不是裸 prefill，而是 `generate(max_new_tokens=1)`

`src/exp_c/bench_prefill_hf.py` 的核心路径是：

- 构造长度为 `4K/8K/16K` 的输入
- 调用 `hf_generate(..., max_new_tokens=1)`
- `TTFT` 取整次 `generate` 的 wall-clock
- `peak_memory_gb` 取 `torch.cuda.max_memory_allocated`

这意味着 `C1` 测到的是：

- prefill 主体计算
- generation 框架开销
- cache 初始化/更新
- 最后一层 `lm_head`
- logits 张量分配

而不是只测 backbone 的 prefill。

### 2.2 `C2` 测的是短 prompt decode，不是长上下文 decode

`src/exp_c/bench_decode_hf.py` 默认：

- `prompt_len=256`
- 实际 prompt 长度约 `240/241`
- `output_len=256/512/1024`
- `TPOT = (total_ms - first_token_ms) / (new_tokens - 1)`

这对“每步 decode 是否依赖上下文长度”很敏感，但**只在 prompt 真够长时才有意义**。当前 `~240 token` 的 prompt 太短，所以它主要测到的是：

- 单 token 路径的常数项
- 每层 kernel 成熟度
- 层数差异

而不是“长上下文下谁的 decode scaling 更好”。

---

## 3. 远程环境与 system 背景

本次远程核查时间：`2026-06-07`  
远程主机：`Chen`，hostname 为 `ubuntu49`

### 3.1 GPU 与 driver

- `8 x NVIDIA GeForce RTX 4090`
- `driver 535.104.05`
- 每卡显存 `24564 MiB`

这意味着：

- 当前机器确实受较老 driver 约束
- 你的 `CUDA 12.1` 兼容性判断是合理的
- 一些更新的 wheel / kernel 路径未必能无痛升级

### 3.2 `nlp` 环境

- Python `3.11.15`
- `torch 2.5.1`
- `torch.version.cuda = 12.1`
- `transformers 4.49.0`
- `datasets 4.8.5`
- `accelerate 1.13.0`
- `huggingface_hub 0.36.2`
- `tokenizers 0.21.4`
- `mamba_ssm 2.2.4`
- `causal_conv1d 1.6.2.post1`
- `triton 3.1.0`

### 3.3 `vllm` 环境

- Python `3.11.15`
- `torch 2.5.1`
- `transformers 4.49.0`
- `vllm 0.7.3`
- `triton 3.1.0`
- `flash_attn` 未安装

### 3.4 和当前官方最新版本的对比

按 PyPI 在 `2026-06-07` 可见的信息：

- `transformers` 最新为 `5.10.2`，发布时间 `2026-06-04`
- `vllm` 最新为 `0.22.1`，发布时间 `2026-06-05`
- `mamba-ssm` 最新为 `2.3.2.post1`，发布时间 `2026-05-09`
- `causal-conv1d` 最新为 `1.6.2.post1`，发布时间 `2026-05-09`

因此当前环境的状态是：

- `causal_conv1d` 并不旧，已经是当前最新版
- 真正偏旧的是 `transformers 4.49.0`、`vllm 0.7.3`、`mamba_ssm 2.2.4`

这点很重要，因为它说明：

- “conv1d kernel 太旧”不是主因
- 更可疑的是 `FalconMamba + transformers 4.49.0 + mamba_ssm 2.2.4` 这一整条实现链

---

## 4. 模型结构层面的关键差异

### 4.1 Llama 的 cache 真的是随上下文线性长

Llama 配置：

- `num_hidden_layers = 32`
- `hidden_size = 4096`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`
- `head_dim = 128`

因此单 token KV cache 大小是：

- 每层：`2 (K,V) * 8 * 128 * 2 bytes = 4096 bytes`
- 全模型：`32 * 4096 = 131072 bytes/token = 128 KiB/token`

所以：

- `4096 token -> 0.536870912 GB`
- `8192 token -> 1.073741824 GB`
- `16384 token -> 2.147483648 GB`

我做的拆解复测里，Llama 的 `forward(use_cache=True)` 返回 cache 大小**和上面完全一致**，说明它的 KV cache 行为是正常的。

### 4.2 Mamba 的 recurrent state 确实是固定大小

Falcon Mamba 配置：

- `num_hidden_layers = 64`
- `hidden_size = 4096`
- `intermediate_size = 8192`
- `state_size = 16`
- `conv_kernel = 4`

HF 的 `MambaCache` 维护两类状态：

- `conv_states`: `[layer, batch, intermediate_size, conv_kernel]`
- `ssm_states`: `[layer, batch, intermediate_size, state_size]`

在当前运行时里：

- `conv_states` 是 `bfloat16`
- `ssm_states` 是 `float32`

实测整个模型的 cache 总大小固定在约 `0.03775 GB`，和上下文长度无关。

这说明你的直觉是对的：

> “Mamba prefill 的 state 不是固定大小吗？”

答案是：

- **是的，recurrent state 是固定大小**
- 但**你在 C1 里看到的 peak memory 不是这个 state 本身**

---

## 5. 为什么 Mamba 的 prefill memory 会随上下文增长，而且比 Llama 还快

这是本次调查里最关键的问题。结论是：**增长的不是 Mamba cache，而是别的东西。**

### 5.1 直接复测：Mamba cache 很小，增长来自别处

我在 Chen 上做了额外拆解，分别测：

- `forward(use_cache=False)`
- `forward(use_cache=True)`
- `generate(max_new_tokens=1)`

#### Llama 复测

| ctx | mode | elapsed ms | peak GB | cache GB |
|---|---:|---:|---:|---:|
| 4096 | forward no cache | 1049.33 | 17.15 | - |
| 4096 | forward cache | 459.26 | 17.69 | 0.5369 |
| 4096 | generate1 | 467.56 | 17.63 | - |
| 8192 | forward no cache | 996.29 | 18.77 | - |
| 8192 | forward cache | 997.50 | 19.85 | 1.0737 |
| 8192 | generate1 | 951.45 | 19.19 | - |
| 16384 | forward no cache | 2192.63 | 21.48 | - |
| 16384 | forward cache | 2245.58 | 23.63 | 2.1475 |
| 16384 | generate1 | 2100.18 | 22.32 | - |

#### Mamba 复测

| ctx | mode | elapsed ms | peak GB | cache GB |
|---|---:|---:|---:|---:|
| 4096 | forward no cache | 573.90 | 18.37 | - |
| 4096 | forward cache | 588.69 | 18.53 | 0.0377 |
| 4096 | generate1 | 560.29 | 16.59 | - |
| 8192 | forward no cache | 1086.82 | 18.11 | - |
| 8192 | forward cache | 1112.30 | 18.39 | 0.0377 |
| 8192 | generate1 | 1113.57 | 18.50 | - |
| 16384 | forward no cache | 2227.46 | 21.56 | - |
| 16384 | forward cache | 2311.80 | 22.10 | 0.0377 |
| 16384 | generate1 | 2312.49 | 22.37 | - |

这组数据直接说明：

- Llama 的 cache 会随上下文线性长，且幅度很大
- Mamba 的 cache 基本固定，只有 `~37.75 MB`
- 但 Mamba 的总峰值显存依然会随上下文长很多

所以，Mamba 的增长来源一定是：

- 中间激活
- selective scan / conv 路径工作区
- hidden states
- logits

### 5.2 真正的“罪魁祸首”之一：Mamba 当前 forward 会做整段 `lm_head`，而且是 `fp32`

`transformers==4.49.0` 中：

- `LlamaForCausalLM.forward` 支持 `logits_to_keep`
- `GenerationMixin` 会在 generation 时自动设 `logits_to_keep=1`
- 但 `FalconMambaForCausalLM.forward` 不支持这个参数
- 且其实现是：

```python
logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
```

也就是说，Mamba 在 prefill / first-token generation 阶段会：

1. 对**整段** hidden states 做 `lm_head`
2. 再把 logits 转成 `float32`

这会造成非常大的线性内存和计算开销。

#### 在 16K 上，这个 logits 张量有多大

- Llama 若算整段 logits：
  - shape 约 `[1, 16384, 128256]`
  - bf16 下约 `4.20 GB`
- Mamba 当前实现：
  - shape 约 `[1, 16384, 65024]`
  - 但它是 `fp32`
  - 大小约 `4.26 GB`

所以虽然 Mamba vocab 更小，但因为它把整段 logits 转成 `fp32`，最终这部分内存几乎和 Llama 的整段 logits 一样大。

### 5.3 为什么 Llama 在你的 C1 里反而没这么惨

因为 Llama 的 generation 路径被 `logits_to_keep=1` 优化掉了。

我单独在 `16K` 上验证了这点：

| setting | elapsed ms | peak GB | logits shape |
|---|---:|---:|---|
| `logits_to_keep=0` | 2816.66 | 22.55 | `[1, 16384, 128256]` |
| `logits_to_keep=1` | 2663.07 | 20.17 | `[1, 1, 128256]` |

这说明：

- 只改这一个参数，Llama 峰值显存就能少约 `2.38 GB`
- 这正是为什么你看到的 `C1` 里，Llama 的 prefill memory 增长没有 Mamba 那么离谱

### 5.4 还有一个因素：Mamba 本身更深，而且 prefill 会走长序列 scan

即使不看 logits，Mamba 也不是“纯固定成本”：

- 它有 `64` 层，不是 `32` 层
- 每层都要做 `in_proj -> conv -> x_proj -> dt_proj -> selective scan -> out_proj`
- 这些步骤在 prefill 阶段都对整个序列生效

所以：

- **固定的是 state，不是 prefill 全流程的 activation/workspace**
- 你的实验结果和这一点是相符的

---

## 6. 为什么 prefill TTFT 两个模型都几乎线性，而不是 Transformer 看起来平方

这里要把“理论复杂度”和“你实际测到的对象”分开。

### 6.1 你测到的是 end-to-end HF generation 路径，不是裸 attention kernel

`C1` 的 TTFT 不是单测 attention，而是：

- embedding
- QKV / MLP / norm
- attention 或 selective scan
- cache 处理
- `lm_head`
- generation 框架逻辑

其中很多项本来就是线性的。

### 6.2 Llama 在当前栈上走的是 PyTorch SDPA，而不是朴素 attention

远程核查显示：

- Llama 配置里 `_attn_implementation = "sdpa"`
- `torch.backends.cuda.flash_sdp_enabled() == True`
- `torch.backends.cuda.mem_efficient_sdp_enabled() == True`

PyTorch 官方文档说明 `scaled_dot_product_attention` 在 CUDA 上会自动选择 fused 实现，包括：

- FlashAttention
- Memory-Efficient Attention
- PyTorch C++ 实现

这意味着：

- Llama 的 attention 没有走“显式构造完整 attention matrix”的低效路径
- 它的 wall-clock 不会像教科书里那样直接呈现一个很干净的平方曲线

### 6.3 更关键的是：当前 benchmark 里，线性项把平方项盖住了

对 Llama 来说：

- 若 generation 路径启用了 `logits_to_keep=1`，就少掉了整段 logits 的巨大成本
- 但其余 `MLP / norm / QKV` 等仍然是线性的
- 在 `4K -> 8K -> 16K` 这个范围内，端到端总时间就可能呈现“近似线性增长”

对 Mamba 来说：

- 它本来就没有 Transformer attention 的平方项
- 再叠加“整段 logits + fp32”的线性成本
- 曲线自然也会近似线性

### 6.4 正确解读

所以这里不能说：

> “Transformer prefill 其实不是平方。”

更准确的说法应该是：

> 在你当前这套 HF + SDPA + generation benchmark 口径下，TTFT 被多种线性项和 fused-kernel 行为主导了，因此**端到端曲线**看起来近似线性，不能直接拿来代表 attention 子模块本身的渐近复杂度。

---

## 7. 为什么 decode TPOT 上 Mamba 反而比 Llama 慢

这是第二个关键问题。结论是：**Mamba 的 decode scaling 更平，但常数项更差。**

### 7.1 先看你原始 `C2` 的结果

`results/exp_c/c2_decode_summary.csv`：

- Llama：`~28.1 ms/token`
- Mamba：`~73.4 ms/token`

如果只看这组数，确实会觉得“这不对，Mamba 不是应该更适合 decode 吗？”

### 7.2 但当前 `C2` prompt 太短，没打到 Mamba 的优势区

我额外做了一个 prompt length sweep，固定 `output_len=128`，测 `prompt_len = 256 / 4K / 8K / 16K`。

#### 额外 decode sweep

| model | prompt 256 | prompt 4K | prompt 8K | prompt 16K |
|---|---:|---:|---:|---:|
| Llama TPOT ms | 28.16 | 28.32 | 32.91 | 46.58 |
| Mamba TPOT ms | 72.73 | 72.87 | 73.26 | 73.34 |

这组数据非常关键：

- **Mamba TPOT 几乎不随 prompt 长度变化**
- **Llama TPOT 会随着 prompt 变长而变慢**

也就是说：

- Mamba 的“长上下文 decode scaling 更好”这件事，在数据上是成立的
- 但在 `16K` 以内、这套实现、这张 4090 上，它的**常数项仍然太大**

### 7.3 为什么 Mamba 的常数项这么大

我认为有四个主要原因。

#### 原因 A：层数翻倍

Llama 是 `32` 层，Falcon Mamba 是 `64` 层。

decode 时虽然 Mamba 不需要像 attention 那样扫完整个 KV cache，但它每生成一个 token，仍然要串行穿过 `64` 个 block。只看层数，它就先天吃亏。

#### 原因 B：单 token 路径并不“只有一个小状态更新”

在当前 fast path 下，Mamba 每层 decode 仍然要做：

- `in_proj`
- `causal_conv1d_update`
- `x_proj`
- `dt_proj`
- `selective_state_update`
- `out_proj`

这些操作对于 `bs=1, seq=1` 的 decode 来说：

- kernel 很碎
- launch 开销更显著
- 算子融合程度有限

所以理论上的“状态固定大小”，并不会自动转化成更低的单 token latency。

#### 原因 C：PyTorch SDPA 路径太成熟，Llama 享受了更多底层优化

Llama 走的是：

- `transformers` 主线支持最好的一类模型
- PyTorch SDPA / FlashAttention 风格后端
- Ada GPU 上已经被广泛打磨过的 attention kernel

而 Falcon Mamba 走的是：

- `transformers 4.49.0` 中的相对新路径
- `mamba_ssm 2.2.4`
- `causal_conv1d + selective_scan` 自定义 kernel 链

两边的工程成熟度并不对等。

#### 原因 D：当前栈偏旧，Mamba 路径更容易吃亏

当前环境里：

- `transformers 4.49.0` 明显落后于当前官方最新版本
- `mamba_ssm 2.2.4` 也落后于当前最新版 `2.3.2.post1`
- `vllm 0.7.3` 也很旧

此外，Falcon Mamba 导入时还会触发：

- `torch.cuda.amp.custom_fwd`
- `torch.cuda.amp.custom_bwd`

的 deprecation warning。

这不是性能证据本身，但它说明：

- 这条代码路径与当前 torch API 的耦合已经偏旧
- “实现没有跟上当前栈”这个判断是合理的

### 7.4 正确解读

所以 `C2` 应该这么读：

- 当前实验**没有否定** Mamba 在长上下文 decode 上的 scaling 优势
- 它只是说明：在 `prompt≈240` 这种很短的设定下，Mamba 的**常数项和实现成熟度劣势**远大于其渐近复杂度优势
- 我额外做的 sweep 也说明了：Mamba 的 TPOT 很平，Llama 的 TPOT 会随着 prompt 长度变大而上升

---

## 8. `C3` 的 system 结果是否合理

`C3` 比的是 `Llama HF vs Llama vLLM`，结果如下：

- `8K` 下 HF 到 `bs=4` OOM，而 vLLM 可以到 `bs=16`
- `16K` 下 HF 到 `bs=2` OOM，而 vLLM 仍可以到 `bs=16`
- vLLM 的 `requests/s` 在更大 batch 上显著高于 HF

这个结果是合理的，原因包括：

1. **vLLM 的核心目标就是 serving，不是单请求 eager inference**
2. 它使用 PagedAttention 风格的 KV 管理，减少 KV cache 碎片和浪费
3. 它的调度设计更适合连续批处理和高并发
4. HF 这边本质上还是“单机 eager generate 循环”

所以 `C3` 的结论是稳的：

- 对 Llama 这种 Transformer 模型，system 层优化会极大改变可用 batch 和吞吐
- “模型理论能不能长”与“系统能不能高效地把长上下文跑起来”是两回事

---

## 9. 对当前 `exp_c` 数据的最终判断

### 9.1 哪些观察是“真实模型现象”

- Llama 的 KV cache 确实随上下文长度线性增长
- Mamba 的 recurrent state 确实基本固定
- Mamba 的 decode TPOT 对 prompt 长度不敏感，scaling 更平
- vLLM 对 Llama serving 的 batch 容量和吞吐提升非常明显

### 9.2 哪些观察带有明显的实现偏置

- `C1` 中 Mamba prefill memory 比 Llama 涨得更快
- `C1` 中两者 TTFT 都近似线性
- `C2` 中 Mamba decode 明显慢于 Llama

这些现象并不是“模型理论”一层就能解释完，而是明显混入了：

- HF generation 口径
- `logits_to_keep` 支持不对称
- Falcon Mamba 整段 `lm_head + fp32 logits`
- `transformers / mamba_ssm` 版本偏旧
- Mamba 自定义 kernel 链成熟度不足

### 9.3 所以这组结果能不能用

可以用，但报告里必须写清楚边界：

- **不要**把 `C1` 直接写成“Llama prefill 比 Mamba 更省显存，所以 Mamba 长上下文优势不成立”
- **不要**把 `C2` 直接写成“Mamba decode 比 Transformer 更慢，所以理论不成立”
- 应该写成：
  - 在当前 `HF 4.49.0` 实现与 CUDA 12.1/4090 环境下，Falcon Mamba 的理论优势没有自然兑现为更好的端到端 TTFT / TPOT
  - 其中一部分是模型深度与常数项问题，一部分是 HF 实现路径问题，一部分是 kernel / system 成熟度问题

---

## 10. 建议你如何修改论文/报告里的叙述

建议把 `exp_c` 写成下面这条叙事：

> 长上下文系统效率不是由“名义复杂度”单独决定的。  
> 在我们的实验中，Mamba 的 recurrent state 确实固定，decode 对上下文长度的敏感性也更低；但在现有 Hugging Face 实现、旧版本 Mamba kernel 栈与单卡 4090 环境下，这些理论优势并未自动转化为更好的端到端 TTFT、显存曲线和 TPOT。相反，生成框架中的 logits 计算策略、模型层数、kernel 成熟度与 serving 系统优化，会显著改变最终观测结果。

这比“谁更线性 / 谁更平方”更有说服力，也更符合你们这门课项目想表达的主线。

---

## 11. 如果要让 `exp_c` 更公平，下一步该怎么改

### 11.1 对 `C1`：把 prefill 拆成 backbone 和 LM head

建议至少新增两组测量：

1. backbone-only prefill
   - Llama：调用 `model.model(...)`
   - Mamba：调用 `model.backbone(...)`

2. last-logit-only prefill
   - Llama：`logits_to_keep=1`
   - Mamba：手工对 `hidden_states[:, -1:, :]` 过 `lm_head`

这样你才能把：

- attention / scan 主体
- cache 开销
- logits 开销

真正拆开。

### 11.2 对 `C2`：扫 `prompt_len`，不要只扫 `output_len`

建议至少测：

- `prompt_len = 256 / 4K / 8K / 16K`
- `output_len = 128` 或 `256`

因为 Mamba 的 decode 优势本来就是“随着 prompt 变长才体现”的。

### 11.3 对 `system`：把版本问题单独写成 limitation

建议在报告里明确写：

- 当前 `transformers / vllm / mamba_ssm` 版本并非最新
- 受 `driver 535 + CUDA 12.1` 约束，不能随意升级到更新栈
- 因此 `exp_c` 结论代表的是“当前可部署软件栈下的端到端表现”，不是架构理论上限

---

## 12. 参考来源

### 远程实测

- Chen 机器环境核查，`2026-06-07`
- 额外拆解复测：
  - `forward(use_cache=False / True / generate1)`
  - decode `prompt_len` sweep

### 官方文档 / 公开来源

- PyPI `mamba-ssm`: <https://pypi.org/project/mamba-ssm/>
- PyPI `causal-conv1d`: <https://pypi.org/project/causal-conv1d/>
- PyPI `transformers`: <https://pypi.org/project/transformers/>
- PyPI `vllm`: <https://pypi.org/project/vllm/>
- Hugging Face FalconMamba 文档: <https://huggingface.co/docs/transformers/model_doc/falcon_mamba>
- PyTorch SDPA 教程: <https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>
- vLLM Paged Attention 文档: <https://docs.vllm.ai/en/latest/design/paged_attention.html>

