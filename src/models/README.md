# Source Model Layer

本目录放源码级模型实现和推理时方法实现。

当前目录：

- `llama/`
- `mamba2/`
- `methods/`

当前状态：

| 对象 | 状态 |
|---|---|
| Llama | 已写本地 PyTorch 架构骨架；权重映射和 tokenizer 绑定待补 |
| Mamba-2 | 占位实现，待补 kernel 和 checkpoint 对接 |
| YaRN / Self-Extend / KIVI / SnapKV / StreamingLLM | 方法占位，待补实现 |
| FIER | 方法占位，待确认具体实现 |

参考源码：

| 对象 | 推荐源码 |
|---|---|
| Mamba-2 | `https://github.com/state-spaces/mamba` |
| Self-Extend | `https://github.com/datamllab/LongLM` |
| KIVI | `https://github.com/jy-yuan/KIVI` |
| SnapKV | `https://github.com/FasterDecoding/SnapKV` |
| StreamingLLM | `https://github.com/mit-han-lab/streaming-llm` |
