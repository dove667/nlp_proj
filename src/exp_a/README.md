# Exp A: Retrieval Baseline

`Exp A` 分成三步：

1. 用 [gen_pred_ruler.py](gen_pred_ruler.py) 生成 `pred.jsonl`
2. 用 [analyze_ruler_predictions.py](analyze_ruler_predictions.py) 做全局评估和位置敏感性分析
3. 用 [plot_position_sensitivity.py](plot_position_sensitivity.py) 生成折线图和热力图

任务：

- 数据来源：`RULER`
- 子任务：`niah_single_1`、`niah_multikey_1`
- 长度档位：Llama 为 4K、8K、16K、32K；Falcon3-Mamba 当前稳定结果为 4K、8K、16K

### `niah_single_1` / `niah_multikey_1` 这两个任务在做什么

这两个子任务都属于 RULER 的 NIAH（needle-in-a-haystack）检索测试，但干扰形式和检索难度不同。

- `niah_single_1`：在长上下文里插入一条目标 key-value 信息，然后询问这条唯一目标对应的 value。这个任务更接近“单针检索”，重点看模型能否在很长的上下文中把唯一目标找出来。
- `niah_multikey_1`：在长上下文里插入多组 key-value 信息，再给定其中一个 key，让模型返回对应的 value。这个任务不只是找到 needle，还要求在多组候选里完成正确的 key-value 绑定，因此比 `single_1` 更容易受到干扰。

可以把两者理解成：

- `niah_single_1`：长文本里找唯一目标
- `niah_multikey_1`：长文本里在多组候选中找对目标并完成绑定

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
  --data_root /data1/zsh/datasets/ruler/niah/Falcon3-Mamba-7B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_a/mamba \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --device_map none \
  --model_device cuda:0 \
  --apply_chat_template \
  --resume
```

说明：

- `Llama` / `Zamba` 可以优先尝试 `--device_map auto`
- `Falcon3-Mamba-7B-Instruct` 如果 `device_map=auto` 有问题，改用 `--device_map none --model_device cuda:0`
- `Falcon3-Mamba-7B-Instruct` 在当前 24GB 4090 环境下，32K 单卡会 OOM；尝试多卡 `device_map=auto` 又会在 Mamba CUDA kernel 路径上报错，所以正式实验只保留到 16K
- 如果设置了 `CUDA_VISIBLE_DEVICES=1`，那么脚本里的 `cuda:0` 就对应物理 `GPU 1`

## Step 2: 分析预测结果

```bash
python src/exp_a/analyze_ruler_predictions.py \
  --data_root /data1/zsh/datasets/ruler/niah/Llama-3.1-8B-Instruct \
  --pred_root results/exp_a/llama31 \
  --lengths 4096 8192 16384 32768 \
  --tasks niah_single_1 niah_multikey_1 \
  --num_bins 10

python src/exp_a/analyze_ruler_predictions.py \
  --data_root /data1/zsh/datasets/ruler/niah/Falcon3-Mamba-7B-Instruct \
  --pred_root results/exp_a/mamba \
  --lengths 4096 8192 16384 \
  --tasks niah_single_1 niah_multikey_1 \
  --num_bins 10
```

输出文件：

- `*.pred.jsonl`：逐样本预测
- `summary.csv`：按长度与任务的全局汇总
- `position_sensitivity_10bins.csv`：10 桶位置敏感性分析

## Step 3: 可视化

使用 [plot_position_sensitivity.py](plot_position_sensitivity.py) 生成位置敏感性可视化图。

```bash
python src/exp_a/plot_position_sensitivity.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_a/llama31/summary.csv \
  --position_csv /data1/zsh/nlp_proj/results/exp_a/llama31/position_sensitivity_10bins.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_a/llama31/position_sensitivity_llama31 \
  --title "Llama-3.1-8B-Instruct on RULER NIAH"

python src/exp_a/plot_position_sensitivity.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_a/mamba/summary.csv \
  --position_csv /data1/zsh/nlp_proj/results/exp_a/mamba/position_sensitivity_10bins.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_a/mamba/position_sensitivity_mamba \
  --title "Falcon3-Mamba-7B-Instruct on RULER NIAH"
```

输出文件：

- `*_line.png`：折线图，适合看不同上下文长度在不同位置桶上的趋势变化
- `*_heatmap.png`：热力图，适合快速定位少数掉点的长度和位置区间

当前 `llama31` 的默认输出示例：

- `results/exp_a/llama31/position_sensitivity_llama31_line.png`
- `results/exp_a/llama31/position_sensitivity_llama31_heatmap.png`

建议：

- 如果整体分数接近满分，优先看热力图，更容易发现稀疏的低分 bucket
- 如果需要比较不同长度随位置变化的走势，再看折线图
- 如果后续补齐 `mamba`结果，直接替换 `summary_csv`、`position_csv` 和 `output_prefix` 即可复用同一脚本

