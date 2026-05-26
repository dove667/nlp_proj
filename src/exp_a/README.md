# Exp A: Retrieval Baseline

目标：按架构比较基础 retrieval 能力，覆盖 NIAH 和 RULER retrieval 子任务。

需要在服务器侧补齐：

- `data_loader.py`：读取已经准备好的 NIAH / RULER retrieval 数据。
- `model_adapter.py`：根据 `model_path` 加载 HF / 官方模型代码。
- `evaluator.py`：按任务输出 accuracy，并汇总有效上下文长度。

入口：

```bash
conda run -n AI python scripts/run_exp_a.py --config configs/exp_a_retrieval.yaml
```
