# Exp B

`Exp B` 现在拆成两个定义不同的子实验：

- `RULER reasoning`：合成推理任务，保留 `4K / 8K / 16K / 32K` 长度扩展，用来测 `reasoning with context scaling`
- `LongBench`：真实下游任务，只跑原始 benchmark 样本，不做人为长度分档，也不做 head-tail 截断

这样做的原因是：

- `RULER reasoning` 是合成控制实验，随着长度增加，答案信息仍然在上下文里，适合分析“长度扩展下的推理退化”
- `LongBench` 的关键证据可能在文档中部。如果先做长度切片或保留头尾再评测，分数会同时受到模型能力和截断策略影响，归因不干净

## Part 1: RULER reasoning

### `vt` / `cwe` / `fwe` 这三个任务在做什么

这三个子任务都属于 RULER 的合成 reasoning 测试，但它们要求模型完成的操作并不一样。

- `vt`：`variable tracking`。输入里会埋入一串变量赋值链，例如 `VAR B = VAR A`、`VAR C = VAR B`。模型需要沿着这些赋值链一直追踪下去，最后找出“哪些变量最终等于某个目标值”。从任务本质上看，`vt` 测的是长上下文中的符号追踪和变量绑定；最终输出通常是一组变量名列表。
- `cwe`：`common words extraction`。输入是一大段编号词表，其中某些普通英文单词会重复出现。模型需要在整段列表中统计频次，并找出出现次数最多的前 `10` 个词。这个任务更像长上下文里的计数与排序；最终输出通常是一组高频自然词。
- `fwe`：`frequent words extraction`。输入是一段由伪词和填充符号混合组成的 coded text。模型需要跟踪每个 coded word 的出现次数，并找出频率最高的前 `3` 个词。它和 `cwe` 一样都涉及频次统计，但 `fwe` 的对象是编码词而不是自然词，最终输出通常是一组短的 coded word 列表。

可以把三者理解成：

- `vt`：顺着赋值链做追踪
- `cwe`：在普通词表里做高频词统计
- `fwe`：在编码文本里做高频词统计

### 生成预测

```bash
python src/exp_b/gen_pred_ruler_reasoning.py \
  --model_path /data1/zsh/models/Llama-3.1-8B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/llama31 \
  --ruler_data_root /data1/zsh/datasets/ruler/reasoning/Llama31_8B_Instruct \
  --ruler_lengths 4096 8192 16384 32768 \
  --ruler_tasks vt cwe fwe \
  --max_new_tokens 0 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --apply_chat_template \
  --resume

python src/exp_b/gen_pred_ruler_reasoning.py \
  --model_path /data1/zsh/models/Falcon3-Mamba-7B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/mamba \
  --ruler_data_root /data1/zsh/datasets/ruler/reasoning/Falcon3-Mamba-7B-Instruct \
  --ruler_lengths 4096 8192 16384 \
  --ruler_tasks vt cwe fwe \
  --max_new_tokens 0 \
  --dtype bf16 \
  --device_map none \
  --model_device cuda:0 \
  --apply_chat_template \
  --resume
```

说明：

- `vt` 使用项目内的 answer-only prompt，并把默认 `max_new_tokens` 设为 `64`
- `cwe` 默认 `120`
- `fwe` 默认 `50`
- 如果显式传入 `--max_new_tokens > 0`，则统一覆盖任务默认值
- 这里接受“开放式 CoT vs 结构化输出”的权衡：自由 CoT 可能帮助模型显式展开推理，但也会显著拉长 decode，并把最终答案推迟到更靠后的 token；对 `vt`，本项目优先选择更短、更稳定、可评测的答案式输出

### 分析预测结果

```bash
python src/exp_b/analyze_ruler_reasoning.py \
  --pred_root /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/llama31 \
  --ruler_lengths 4096 8192 16384 32768 \
  --ruler_tasks vt cwe fwe

python src/exp_b/analyze_ruler_reasoning.py \
  --pred_root /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/mamba \
  --ruler_lengths 4096 8192 16384 \
  --ruler_tasks vt cwe fwe
```

输出文件：

- `summary.csv`：每个 task / length 的汇总分数

### 可视化

使用 [plot_ruler_reasoning.py](plot_ruler_reasoning.py)：

```bash
python src/exp_b/plot_ruler_reasoning.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/llama31/summary.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/llama31/ruler_reasoning \
  --title "Llama-3.1-8B-Instruct on RULER Reasoning"

python src/exp_b/plot_ruler_reasoning.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/mamba/summary.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/mamba/ruler_reasoning \
  --title "Falcon3-Mamba-7B-Instruct on RULER Reasoning"
```

## Part 2: LongBench

这一部分不做人为长度分档，也不再做 head-tail 截断。LongBench 在本项目中的定义是：

- 直接对原始 benchmark 样本做推理
- 不再尝试构造 `4K / 8K / 16K / 32K` 的伪长度曲线
- 结果解释聚焦于真实任务表现，而不是 synthetic context scaling

### 生成预测

```bash
python src/exp_b/gen_pred_longbench.py \
  --model_path /data1/zsh/models/Llama-3.1-8B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_b/longbench/llama31 \
  --longbench_data_root /data1/zsh/datasets/LongBench \
  --longbench_tasks hotpotqa qasper gov_report repobench-p \
  --max_new_tokens 0 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --apply_chat_template \
  --resume

python src/exp_b/gen_pred_longbench.py \
  --model_path /data1/zsh/models/Falcon3-Mamba-7B-Instruct \
  --out_root /data1/zsh/nlp_proj/results/exp_b/longbench/mamba \
  --longbench_data_root /data1/zsh/datasets/LongBench \
  --longbench_tasks hotpotqa qasper gov_report repobench-p \
  --max_new_tokens 0 \
  --dtype bf16 \
  --device_map none \
  --model_device cuda:0 \
  --apply_chat_template \
  --resume
```

LongBench 的项目内默认生成长度：

- `hotpotqa=32`
- `qasper=64`
- `gov_report=256`
- `repobench-p=128`

这些值不是官方统一标准，而是按任务输出形态给出的更稳妥推理预算。

### 分析预测结果

```bash
python src/exp_b/analyze_longbench.py \
  --pred_root /data1/zsh/nlp_proj/results/exp_b/longbench/llama31 \
  --longbench_tasks hotpotqa qasper gov_report repobench-p

python src/exp_b/analyze_longbench.py \
  --pred_root /data1/zsh/nlp_proj/results/exp_b/longbench/mamba \
  --longbench_tasks hotpotqa qasper gov_report repobench-p
```

输出文件：

- `summary.csv`：每个 task 的汇总分数

### 可视化

```bash
python src/exp_b/plot_longbench.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_b/longbench/llama31/summary.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_b/longbench/llama31/longbench \
  --title "Llama-3.1-8B-Instruct on LongBench"

python src/exp_b/plot_longbench.py \
  --summary_csv /data1/zsh/nlp_proj/results/exp_b/longbench/mamba/summary.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_b/longbench/mamba/longbench \
  --title "Falcon3-Mamba-7B-Instruct on LongBench"
```
