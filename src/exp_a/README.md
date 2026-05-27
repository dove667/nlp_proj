# Exp A: Retrieval Baseline

目标：按架构比较基础 retrieval 能力，覆盖 NIAH 和 RULER retrieval 子任务。

需要在服务器侧补齐：

- `data_loader.py`：读取已经准备好的 NIAH / RULER retrieval 数据。
- `model_adapter.py`：根据 `model_path` 加载 HF / 官方模型代码。
- `evaluator.py`：按任务输出 accuracy，并汇总有效上下文长度。

入口：

```bash
python /data1/zsh/eval_ruler_hf.py \
--model_path /data1/zsh/models/Llama-3.1-8B-Instruct \
--data_root /data1/zsh/datasets/ruler \
--out_root /data1/zsh/results/ruler \
--lengths 4096 8192 16384 32768 \
--tasks niah_single_1 niah_multikey_1 \
--max_new_tokens 128 \
--dtype bf16 \
--attn_implementation sdpa \
--apply_chat_template \
--resume
```