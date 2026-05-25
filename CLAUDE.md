# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

NLP 课程项目，主题为**大语言模型的长上下文能力拆解与评测**。核心论点：长上下文能力不是单一维度，而是名义窗口长度、有效检索长度、长程推理能力、inference-time 扩窗效果、系统吞吐与延迟共同决定的结果。

当前阶段：文档完整，代码骨架尚未搭建。主要工作是搭实验框架、跑 benchmark、输出图表和报告素材。

## 实验主线

四层实验设计，每层回答一个子问题：

| 实验 | Benchmark | 目的 |
|------|-----------|------|
| A | NIAH + RULER retrieval | 名义窗口 vs 有效检索长度 |
| B | RULER multi-hop/aggregation + LongBench | retrieval vs reasoning |
| C | baseline vs Self-Extend (vs InfLLM) | inference-time 扩窗是否有效 |
| D | HF / vLLM / SGLang 系统测量 | 系统瓶颈（TTFT、TPOT、throughput、peak memory） |

推荐主模型：`Llama-3.1-8B-Instruct` 或 `Qwen2.5-7B-Instruct`（7B/8B 在 8×4090 上做多长度矩阵实验更稳）。

测试长度档位：4K / 8K / 16K / 32K / 64K / 128K。

## 推荐代码目录结构（待搭建）

```
nlp_proj/
├── configs/          # 实验配置文件（YAML）
├── scripts/          # 数据预处理、benchmark 运行、系统测量脚本
├── src/
│   ├── data/         # 数据加载与生成（NIAH、RULER 合成数据）
│   ├── evaluation/   # 各 benchmark 评测逻辑
│   ├── models/       # 模型加载、Self-Extend / InfLLM 适配
│   └── utils/        # 指标计算、结果保存
├── results/          # 实验输出（JSON/CSV + 图表）
└── notebooks/        # 结果可视化
```

## 关键文档

- `docs/experiments.md`：完整实验设计，包含每层实验的参数矩阵、指标定义和推荐图表形式
- `docs/survey1/2/3.md`：长上下文建模综述（架构、位置编码、KV cache、benchmark 对比）
- `report/main.tex`：课程报告 LaTeX 源文件，已有 abstract 和 introduction
- `report/references.bib`：参考文献库
- `AGENTS.md`：面向 AI 助手的项目上下文，包含优先级建议

