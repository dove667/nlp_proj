# Source Model Layer

本目录放模型与推理方法源码，用于把不同模型架构和推理时方法统一接到四个实验。

约定：

- 不使用 `AutoModel` / `AutoModelForCausalLM` 作为模型加载入口。
- 权重、tokenizer、数据集都假设已经在服务器上准备好。
- 每个模型有自己的源码目录：`llama/`、`longformer/`、`mamba2/`、`jamba/`。
- 四个实验目录里仍然保留 experiment-specific adapter，负责把数据集样本转成模型输入。
- 本目录提供统一 factory、结果协议和模型源码。复杂或 checkpoint 强相关的部分用 TODO 标出来。

当前实现状态：

| 对象 | 状态 |
|---|---|
| Llama | 已写 PyTorch 架构骨架：RMSNorm、RoPE、GQA attention、MLP、decoder、causal LM；权重映射和 Llama-3 tokenizer 绑定待补 |
| Longformer | 保留源码类，占位；需根据任务决定 encoder head 或 QA wrapper |
| Mamba-2 | 保留源码类，占位；需要 selective scan kernel 和 checkpoint 命名后补 |
| Jamba | 保留源码类，占位；需要 hybrid/MoE 路由和 checkpoint 格式后补 |
| YaRN / Self-Extend / KIVI / SnapKV / StreamingLLM | 保留方法源码类，占位；要改 Llama attention / KV cache 之后实现 |
| FIER | 未确认具体源码，实现留给后续 |

已确认的源码仓库：

| 对象 | 推荐源码 |
|---|---|
| Longformer | `https://github.com/allenai/longformer` |
| Mamba-2 | `https://github.com/state-spaces/mamba` |
| Self-Extend | `https://github.com/datamllab/LongLM` |
| KIVI | `https://github.com/jy-yuan/KIVI` |
| SnapKV | `https://github.com/FasterDecoding/SnapKV` |
| StreamingLLM | `https://github.com/mit-han-lab/streaming-llm` |

暂未硬接：

- `FIER`：目前先保留 adapter，占位给服务器侧源码实现。
- `Jamba`：开放权重主要通过 AI21 / HF 生态分发，当前只保留本地源码入口。
- `Llama-3.1`：当前已有本地 PyTorch 骨架，但还没有完成 checkpoint tensor name 映射。
