# Exp B: Reasoning Benchmarks

`Exp B` 分成三步：

1. 用 [gen_pred_benchmarks.py](gen_pred_benchmarks.py) 生成 `pred.jsonl`
2. 用 [analyze_benchmarks.py](analyze_benchmarks.py) 汇总各任务分数并计算长度退化
3. 用 [plot_benchmark_results.py](plot_benchmark_results.py) 做结果可视化

任务：

- 数据来源：`RULER reasoning + LongBench`
- RULER reasoning 子任务：`vt`、`cwe`、`fwe`
- LongBench 子集：`hotpotqa`、`qasper`、`gov_report`、`repobench-p`
- 长度档位：`4K / 8K / 16K / 32K`

## Step 1: 生成预测

命令行参数：

- `--model_path`：模型目录
- `--out_root`：输出目录
- `--ruler_data_root`：RULER reasoning 数据根目录
- `--ruler_lengths`：RULER reasoning 长度列表
- `--ruler_tasks`：RULER reasoning 任务列表
- `--longbench_data_root`：LongBench 数据目录
- `--longbench_lengths`：LongBench 目标上下文长度列表
- `--longbench_tasks`：LongBench 任务列表
- `--max_new_tokens`
- `--dtype`
- `--device_map`
- `--model_device`：当 `--device_map none` 时生效
- `--apply_chat_template`
- `--resume`

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/exp_b/gen_pred_benchmarks.py \
  --model_path /data1/zsh/models/Llama-3.1-8B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_b/llama31 \
  --ruler_data_root /data1/zsh/datasets/ruler/reasoning/Llama-3.1-8B-Instruct \
  --ruler_lengths 4096 8192 16384 32768 \
  --ruler_tasks vt cwe fwe \
  --longbench_data_root /data1/zsh/datasets/LongBench \
  --longbench_lengths 8192 16384 32768 \
  --longbench_tasks hotpotqa qasper gov_report repobench-p \
  --max_new_tokens 128 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --apply_chat_template \
  --resume

CUDA_VISIBLE_DEVICES=1 python src/exp_b/gen_pred_benchmarks.py \
  --model_path /data1/zsh/models/Falcon3-Mamba-7B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_b/mamba \
  --ruler_data_root /data1/zsh/datasets/ruler/reasoning/Falcon3-Mamba-7B-Instruct \
  --ruler_lengths 4096 8192 16384 32768 \
  --ruler_tasks vt cwe fwe \
  --longbench_data_root /data1/zsh/datasets/LongBench \
  --longbench_lengths 8192 16384 32768 \
  --longbench_tasks hotpotqa qasper gov_report repobench-p \
  --max_new_tokens 128 \
  --dtype bf16 \
  --device_map none \
  --model_device cuda:0 \
  --apply_chat_template \
  --resume
```

说明：

- `Llama` 可以优先尝试 `--device_map auto`
- `Falcon3-Mamba-7B-Instruct` 建议优先使用 `--device_map none --model_device cuda:0`
- LongBench 在这里按目标长度做 token-budget 截断，默认保留上下文前后两端

## Step 2: 分析预测结果

```bash
python src/exp_b/analyze_benchmarks.py \
  --pred_root /data1/zsh/nlp_proj/results/exp_b/llama31 \
  --ruler_lengths 4096 8192 16384 32768 \
  --ruler_tasks vt cwe fwe \
  --longbench_lengths 4096 8192 16384 32768 \
  --longbench_tasks hotpotqa qasper gov_report repobench-p
```

输出文件：

- `summary.csv`：每个 benchmark / task / length 的汇总分数
- `decay.csv`：相对最短长度的分数变化 `Δ(L_shortest, L_target)`

指标说明：

- `vt / cwe / fwe`：`accuracy`
- `hotpotqa / qasper`：`QA F1`
- `gov_report`：`Rouge-L F1`
- `repobench-p`：`exact match`

## Step 3: 可视化

```bash
micromamba run -n zsh python src/exp_b/plot_benchmark_results.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_b/llama31/summary.csv \
  --decay_csv /data1/zsh/nlp_proj/results/exp_b/llama31/decay.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_b/llama31/benchmark_results \
  --title "Llama-3.1-8B-Instruct on Exp B"
```

输出文件：

- `*_scores.png`：各任务随长度变化的主结果图
- `*_decay.png`：相对最短长度的退化对比图

建议：

- 先看 `scores.png`，判断不同任务的绝对水平和随长度变化趋势
- 再看 `decay.png`，更方便比较 reasoning 任务在长上下文下的退化幅度
