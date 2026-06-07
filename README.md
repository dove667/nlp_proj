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
- `src/exp_c/` — Exp C, reorganized into three benchmark scripts plus one unified analysis script
- `src/data/` — RULER data preparation notes
- `src/utils.py` — Shared I/O and model utilities
- `results/` — Experiment output directory
- `report/` — LaTeX report

## Environment

Experiment environment: Python 3.11 / PyTorch 2.5.1+cu121 / Transformers 4.49.0.

Use `micromamba` / `conda` for the binary-sensitive runtime base: Python, PyTorch, CUDA, Triton.  
Use `uv pip` for upper-layer Python dependencies from [`pyproject.toml`](pyproject.toml).

Main environment:

```bash
micromamba env create -f environment.yml
micromamba activate nlp
uv lock
uv export --frozen --no-dev --format requirements.txt --prune torch --output-file .uv-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -r .uv-requirements.txt
```

Separate `vllm` environment (`cu121`):

```bash
micromamba env create -f environment.vllm.yml
micromamba activate vllm
uv pip install --python "$CONDA_PREFIX/bin/python" vllm==0.7.3
```

Optional Mamba kernels:

```bash
micromamba activate nlp
CUDA_HOME=/usr/local/cuda MAX_JOBS=8 pip install --no-deps --no-build-isolation "causal-conv1d>=1.4.0"
```

`mamba-ssm` local wheel:

```bash
micromamba activate nlp
pip install --no-deps ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Check installation:

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
- **Exp C** now separates HF prefill, HF decode, and Llama HF-vs-vLLM backend comparison

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
- `src/exp_c/` — Exp C，重组为三个 benchmark 脚本和一个统一分析脚本
- `src/data/` — RULER 数据准备说明
- `src/utils.py` — 共享 I/O 与模型工具函数
- `results/` — 实验输出目录
- `report/` — LaTeX 报告

## 环境依赖

实验环境：Python 3.11 / PyTorch 2.5.1+cu121 / Transformers 4.49.0。

底层敏感运行时使用 `micromamba` / `conda` 管理：Python、PyTorch、CUDA、Triton。  
上层 Python 依赖使用 `uv pip` 配合 [`pyproject.toml`](pyproject.toml) 管理。

主环境：

```bash
micromamba env create -f environment.yml
micromamba activate nlp
uv lock
uv export --frozen --no-dev --format requirements.txt --prune torch --output-file .uv-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -r .uv-requirements.txt
```

单独的 `vllm` 环境（`cu121`）：

```bash
micromamba env create -f environment.vllm.yml
micromamba activate vllm
uv pip install --python "$CONDA_PREFIX/bin/python" vllm==0.7.3
```

可选 Mamba kernel：

```bash
micromamba activate nlp
CUDA_HOME=/usr/local/cuda MAX_JOBS=8 pip install --no-deps --no-build-isolation "causal-conv1d>=1.4.0"
```

`mamba-ssm` 本地 wheel：

```bash
pip install --no-deps ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

检查安装：

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
- **Exp C** 现在拆成三部分：HF prefill、HF decode、Llama HF-vs-vLLM backend 对比

模型主线：

- `Meta/Llama-3.1-8B-Instruct`
- `tiiuae/Falcon3-Mamba-7B-Instruct`

详见[实验设计文档](docs/experiments.md)。
