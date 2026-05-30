# NLP Course Project

**[中文版本](#chinese-version)**

This is the repository for a Southern University of Science and Technology NLP course project.  
Topic: **What Determines Long-Context Ability in LLMs? A Survey with Experimental Analysis**.

## Directory Structure

- `docs/survey1.md`, `docs/survey2.md`, `docs/survey3.md` — Literature survey and reading notes
- `docs/experiments.md` — Experiment blueprint
- `docs/report_outline.md` — Report outline
- `src/exp_a/` — Exp A, RULER NIAH retrieval
- `src/exp_b/` — Exp B, split into RULER reasoning (length-extended reasoning) and LongBench (real tasks on original samples)
- `src/exp_c/` — Exp C, serving performance benchmarks (TTFT, TPOT, throughput, peak memory)
- `src/data/` — RULER data preparation notes
- `src/utils.py` — Shared I/O and model utilities
- `results/` — Experiment output directory
- `report/` — LaTeX report

## Environment

Experiment environment: Python 3.11 / PyTorch 2.5.1+cu121 / Transformers 4.49.0, managed by a micromamba env `nlp_proj`.

[`pyproject.toml`](pyproject.toml) is the dependency manifest, installed with `uv pip install`:

```bash
micromamba activate nlp_proj
uv pip install matplotlib pandas vllm  # add as needed
```

`torch` is pinned to the `cu121` index to avoid resolving to an incompatible CUDA version.

`Falcon3-Mamba-7B-Instruct` runs directly with `transformers` — `mamba-ssm` is not a hard dependency. Stable config: single GPU, `bf16`, `--device_map none`. Do **not** use `device_map="auto"`. 32K OOMs on a single GPU; official results exclude Mamba 32K.

### Mamba kernel (optional)

If you need `mamba-ssm`, install from the local wheel to avoid network issues. Verified combination:

- `torch==2.5.1+cu121`
- `causal-conv1d==1.6.2.post1`
- `mamba-ssm==2.2.4`

Install:

```bash
micromamba activate nlp_proj
uv pip install --no-deps ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Verify:

```bash
python - <<'PY'
import torch
print("torch =", torch.__version__, "cuda =", torch.version.cuda)

import causal_conv1d
print("causal_conv1d ok")

import mamba_ssm
print("mamba_ssm ok")
PY
```

## Experiments

- Data sources: `RULER + LongBench`
- **Exp A** uses RULER NIAH subtasks: `niah_single_1`, `niah_multikey_1`
- **Exp B** covers two classes of evaluation: RULER reasoning (reasoning degradation under length extension) and LongBench (raw-sample real-world tasks)
- **Exp C** focuses on TTFT, TPOT, throughput, and peak GPU memory

Model lineup:

- `Meta/Llama-3.1-8B-Instruct`
- `tiiuae/Falcon3-Mamba-7B-Instruct`

See [experiment design doc](docs/experiments.md) for details.

---

<a id="chinese-version"></a>

# NLP 课程项目

本仓库是南方科技大学 NLP 课程项目仓库。  
主题：**What Determines Long-Context Ability in LLMs? A Survey with Experimental Analysis**。

## 目录结构

- `docs/survey1.md`, `docs/survey2.md`, `docs/survey3.md` — 文献综述与阅读笔记
- `docs/experiments.md` — 实验主线
- `docs/report_outline.md` — 报告提纲
- `src/exp_a/` — Exp A，RULER NIAH retrieval
- `src/exp_b/` — Exp B，拆分为 RULER reasoning（长度扩展推理）与 LongBench（原始样本真实任务）
- `src/exp_c/` — Exp C，serving 性能测试（TTFT、TPOT、throughput、peak memory）
- `src/data/` — RULER 数据准备说明
- `src/utils.py` — 共享 I/O 与模型工具函数
- `results/` — 实验输出目录
- `report/` — LaTeX 报告

## 环境依赖

实验环境：Python 3.11 / PyTorch 2.5.1+cu121 / Transformers 4.49.0，由 micromamba env `nlp_proj` 管理。

[`pyproject.toml`](pyproject.toml) 是依赖清单，配合 `uv pip install` 安装：

```bash
micromamba activate nlp_proj
uv pip install matplotlib pandas vllm  # 按需补充
```

`torch` 固定到 `cu121` 源，避免解析到不兼容的 CUDA 版本。

`Falcon3-Mamba-7B-Instruct` 在 `transformers` 中可以直接推理，不强依赖 `mamba-ssm`。常见稳定配置：单卡、`bf16`、`--device_map none`，不要用 `device_map="auto"`。32K 单卡 OOM，正式结果不包含 Mamba 32K。

### Mamba kernel（可选）

如需安装 `mamba-ssm`，用本地 wheel 避免网络问题。已验证的组合：

- `torch==2.5.1+cu121`
- `causal-conv1d==1.6.2.post1`
- `mamba-ssm==2.2.4`

安装：

```bash
micromamba activate nlp_proj
uv pip install --no-deps ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

验证：

```bash
python - <<'PY'
import torch
print("torch =", torch.__version__, "cuda =", torch.version.cuda)

import causal_conv1d
print("causal_conv1d ok")

import mamba_ssm
print("mamba_ssm ok")
PY
```

## 实验

- 数据来源：`RULER + LongBench`
- **Exp A** 使用 RULER 下的 NIAH 子任务：`niah_single_1`、`niah_multikey_1`
- **Exp B** 同时包含两类定义不同的评测：RULER reasoning 做长度扩展下的推理退化分析，LongBench 直接跑原始样本
- **Exp C** 关注 TTFT、TPOT、throughput、peak GPU memory

模型主线：

- `Meta/Llama-3.1-8B-Instruct`
- `tiiuae/Falcon3-Mamba-7B-Instruct`

详见[实验设计文档](docs/experiments.md)。
