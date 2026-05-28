# Exp A: Retrieval Baseline

`Exp A` 分成两步：

1. 用 [gen_pred_ruler.py](gen_pred_ruler.py) 生成 `pred.jsonl`
2. 用 [analyze_ruler_predictions.py](analyze_ruler_predictions.py) 做全局评估和位置敏感性分析

任务：

- 数据来源：`RULER`
- 子任务：`niah_single_1`、`niah_multikey_1`
- 长度档位：4K、8K、16K、32K

## Step 1: 生成预测

命令行参数：

- `--model_path`：模型目录
- `--data_root`：RULER 数据根目录
- `--out_root`：输出目录
- `--lengths`：要跑的长度列表
- `--tasks`：任务列表
- `--max_new_tokens`
- `--dtype`
- `--device_map`
- `--model_device`：当 `--device_map none` 时生效
- `--attn_implementation`
- `--apply_chat_template`
- `--resume`

示例：

```bash
CUDA_VISIBLE_DEVICES=1,2 python src/exp_a/gen_pred_ruler.py \
  --model_path /data1/zsh/models/Llama-3.1-8B-Instruct \
  --data_root /data1/zsh/datasets/ruler/Llama31_8B_Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_a/llama31 \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --apply_chat_template \
  --resume

CUDA_VISIBLE_DEVICES=1 python src/exp_a/gen_pred_ruler.py \
  --model_path /data1/zsh/models/Falcon3-Mamba-7B-Instruct \
  --data_root /data1/zsh/datasets/ruler/Falcon3-Mamba-7B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_a/mamba \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --device_map none \
  --model_device cuda:0 \
  --apply_chat_template \
  --resume

python src/exp_a/gen_pred_ruler.py \
  --model_path /data1/zsh/models/Zamba2-7B-Instruct-v2 \
  --data_root /data1/zsh/datasets/ruler/Zamba2-7B-Instruct-v2 \
  --out_root /data1/zsh/nlp_proj/results/exp_a/zamba \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --apply_chat_template \
  --resume
```

说明：

- `Llama` / `Zamba` 可以优先尝试 `--device_map auto`
- `Falcon3-Mamba-7B-Instruct` 如果 `device_map=auto` 有问题，改用 `--device_map none --model_device cuda:0`
- 如果设置了 `CUDA_VISIBLE_DEVICES=1`，那么脚本里的 `cuda:0` 就对应物理 `GPU 1`

## Step 2: 分析预测结果

```bash
python src/exp_a/analyze_ruler_predictions.py \
  --data_root /data1/zsh/datasets/ruler/Llama31_8B_Instruct \
  --pred_root results/exp_a/llama31 \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --num_bins 10
```

输出文件：

- `*.pred.jsonl`：逐样本预测
- `summary.csv`：按长度与任务的全局汇总
- `position_sensitivity_10bins.csv`：10 桶位置敏感性分析
