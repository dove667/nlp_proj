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

评估指标：

- `RULER reasoning` 当前统一按 `accuracy` 统计。
- 对单条样本，只有当标准答案里的所有目标项都出现在模型输出中时，才记为正确；否则记为错误。
- 因此它本质上是一个基于答案项覆盖的 exact-match 风格判分，而不是部分给分。

### 可视化

使用 [plot_ruler_reasoning.py](plot_ruler_reasoning.py)：

```bash
python src/exp_b/plot_ruler_reasoning.py   --llama_summary_csv /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/llama31/summary.csv   --mamba_summary_csv /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/mamba/summary.csv   --output_prefix /data1/zsh/nlp_proj/results/exp_b/ruler_reasoning/ruler_reasoning   --title "Exp B RULER Reasoning"
```

## Part 2: LongBench

这一部分直接对原始 benchmark 样本做推理，结果解释聚焦于真实任务表现。

### `hotpotqa` / `qasper` / `gov_report` / `repobench-p` 这四个任务在做什么

这四个子任务覆盖了 LongBench 中几种很不一样的长上下文能力。

- `hotpotqa`：多文档问答任务。模型需要从多段材料中找到支持证据，并回答一个通常带有多跳性质的问题。
- `qasper`：学术文档问答任务。模型需要阅读论文或论文片段，理解其中的方法、实验或结论，再给出简洁回答。
- `gov_report`：长文档摘要任务。模型需要从较长的政府报告中提炼关键信息，生成一段覆盖主要结论的摘要。
- `repobench-p`：代码补全任务。模型会拿到一段 repository context 和代码前缀，需要继续补出缺失的代码片段。

可以把四者理解成：

- `hotpotqa`：长上下文里的多跳问答
- `qasper`：长文档阅读后的论文问答
- `gov_report`：长报告摘要
- `repobench-p`：长代码上下文补全

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
  --max_input_tokens 20000 \
  --dtype bf16 \
  --device_map none \
  --model_device cuda:0 \
  --apply_chat_template \
  --resume
```

**Mamba 单卡显存限制说明**

Falcon3-Mamba-7B-Instruct 在 24GB RTX 4090 上，使用 `mamba_ssm` CUDA kernel（`nlp_proj` conda 环境）时，`generate()` 峰值显存随序列长度线性增长约 0.44 GB/1K tokens：

| 序列长度 | 峰值显存 | 状态 |
|----------|----------|------|
| 16000    | 21.62 GB | OK   |
| 18000    | 22.50 GB | OK   |
| 20000    | 23.38 GB | OK   |
| 21000    | —        | OOM  |

安全上限为 **20000 tokens**（含 prompt overhead）。不能用 `device_map="auto"` 多卡分片（Mamba CUDA kernel 路径存在稳定性问题）。

**碎片 OOM 问题**：PyTorch 的 caching allocator 在长序列推理时容易产生内存碎片，导致实际可用连续块不足，即使理论显存够用也会 OOM（报错中会出现 `reserved but unallocated` 远大于申请量的情况）。代码已在启动时设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 解决此问题，无需手动设置。

`--max_input_tokens 20000` 会在 tokenize 之前截断 context 尾部，保留 prompt 结构，并在输出中记录 `context_truncated: true`。各任务受影响情况（基于 Mamba tokenizer 实测）：

| 任务 | 样本数 | 超 20K 条数 | 占比 |
|------|--------|-------------|------|
| hotpotqa | 200 | 0 | 0% |
| qasper | 200 | 2 | 1% |
| gov_report | 200 | 13 | 6% |
| repobench-p | 500 | 80 | 16% |

Llama 不受此限制，不需要传 `--max_input_tokens`。

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

评估指标：

- `hotpotqa`、`qasper`：使用 `QA F1`，比较预测答案和标准答案的词项重合程度。
- `gov_report`：使用 `Rouge-L F1`，衡量生成摘要和参考摘要之间的最长公共子序列重合。
- `repobench-p`：使用 `edit similarity`（LongBench 官方指标），基于 Levenshtein 编辑距离计算 `1 - edit_distance / max(len(pred), len(gold))`。不使用 exact match——模型通常会生成多行代码而标准答案只有一行，exact match 会导致分数虚低为 0。

### 可视化

```bash
python src/exp_b/plot_longbench.py \
  --llama_csv /data1/zsh/nlp_proj/results/exp_b/longbench/llama31/summary.csv \
  --mamba_csv /data1/zsh/nlp_proj/results/exp_b/longbench/mamba/summary.csv \
  --output_prefix /data1/zsh/nlp_proj/results/exp_b/longbench/longbench \
  --title "Exp B LongBench"
```

输出：`results/exp_b/longbench/longbench_compare.pdf`，两个模型并排分组柱状图。
