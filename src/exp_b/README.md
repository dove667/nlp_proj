# Exp B: Reasoning

目标：比较 retrieval 能力强的模型，在 RULER 复杂任务和 LongBench 子集上的长程推理表现。

需要在服务器侧补齐：

- `data_loader.py`：读取 RULER reasoning 和 LongBench 子集。
- `model_adapter.py`：加载四类架构模型。
- `evaluator.py`：按任务选择 accuracy / F1 / ROUGE-L，并计算 accuracy decay slope。

入口：

```bash
conda run -n AI python scripts/run_exp_b.py --config configs/exp_b_reasoning.yaml
```
