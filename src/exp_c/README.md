# Exp C: Inference-Time Extension

目标：在 Llama 主模型上比较 baseline、YaRN、Self-Extend、KIVI、SnapKV、FIER、StreamingLLM 等推理时扩窗/优化方法。

需要在服务器侧补齐：

- `method_adapter.py`：为每种方法包装模型加载、KV cache 策略或推理后端。
- `data_loader.py`：复用 Exp A/B 的 NIAH、RULER、LongBench 数据。
- `evaluator.py`：输出 accuracy gain、有效上下文长度变化和 latency overhead。

入口：

```bash
conda run -n AI python scripts/run_exp_c.py --config configs/exp_c_inference_extension.yaml
```
