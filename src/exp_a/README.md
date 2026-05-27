# Exp A: Retrieval Baseline

`Exp A` 直接运行 [eval_ruler_hf.py](eval_ruler_hf.py)。

任务：

- 数据来源：`RULER`
- 子任务：`niah_single_1`、`niah_multikey_1`
- 长度档位：4K、8K、16K、32K

命令行参数：

- `--model_path`：模型目录
- `--data_root`：RULER 数据根目录
- `--out_root`：输出目录
- `--lengths`：要评测的长度列表
- `--tasks`：任务列表
- `--max_new_tokens`
- `--dtype`
- `--device_map`
- `--attn_implementation`
- `--apply_chat_template`
- `--resume`

示例：

```bash
python src/exp_a/eval_ruler_hf.py \
  --model_path /data1/zsh/models/Llama-3.1-8B-Instruct \
  --data_root /data1/zsh/datasets/ruler \
  --out_root /data1/zsh/results/exp_a/llama31 \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --apply_chat_template \
  --resume
```

评测另外两个模型时，直接替换 `--model_path` 和 `--out_root` 即可，例如：

- `--model_path /data1/zsh/models/Falcon3-Mamba-7B-Instruct`
- `--model_path /data1/zsh/models/Zamba2-7B-Instruct-v2`

输出文件：

- `*.pred.jsonl`：逐样本预测
- `summary.csv`：按长度与任务汇总
