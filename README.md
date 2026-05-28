# NLP Course Project

本仓库是南方科技大学 NLP 课程项目仓库，主题是：**What Determines Long-Context Ability in LLMs? A Survey with Experimental Analysis**。

## 目录结构

- `docs/survey1.md`, `docs/survey2.md`, `docs/survey3.md`：文献综述与阅读笔记
- `docs/experiments.md`：实验主线
- `docs/report_outline.md`：报告提纲
- `src/exp_a/`：Exp A，RULER NIAH retrieval
- `src/exp_b/`：Exp B，RULER reasoning + LongBench
- `src/exp_c/`：Exp C，推理时扩窗 / KV 优化方法
- `src/exp_d/`：Exp D，serving 性能测试
- `src/models/`：模型与方法实现
- `src/data/`：RULER 数据准备说明
- `results/`：实验输出目录
- `report/`：LaTeX 报告

## 环境依赖

当前在服务器上验证通过的实验环境是：

- Python `3.11`
- PyTorch `2.5.1+cu121`
- Transformers `4.49.0`
- Accelerate `1.13.0`
- Triton `3.1.0`

仓库现在提供了一个基于 `uv` 的 [`pyproject.toml`](pyproject.toml)。推荐工作流如下：

```bash
cd /data1/zsh/nlp_proj
conda create -n nlp_proj python=3.11 -y
conda activate nlp_proj
pip install uv
uv sync
```

说明：

- `torch` / `torchvision` / `torchaudio` 已固定到 `cu121` 源，避免解析时误切到不兼容的 CUDA 版本。
- `Falcon3-Mamba-7B-Instruct` 在 `transformers` 中可以直接推理，不强依赖 `mamba-ssm`。

如果要做 Mamba kernel / efficiency 相关实验，可以额外安装可选依赖：

```bash
cd /data1/zsh/nlp_proj
CUDA_HOME=/usr/local/cuda MAX_JOBS=8 uv sync --extra mamba-kernels
```

注意：

- `causal-conv1d` 已放进可选 extra。
- `mamba-ssm` 暂时没有放进基础依赖里，因为它和当前 `torch/triton` 组合仍需要单独验证。
- 对于 `Falcon3-Mamba-7B-Instruct`，常见稳定配置是单卡、`bf16/fp16`、不要使用 `device_map="auto"`。

当前在 `Chen` 上已经验证通过的 Mamba kernel 组合是：

- `torch==2.5.1+cu121`
- `causal-conv1d==1.6.2.post1`
- `mamba-ssm==2.2.4+cu12torch2.5cxx11abifalse`

对于 `mamba-ssm`，更推荐使用“本地下载 wheel，再上传到服务器安装”的方式，尤其是在服务器网络不稳定、GitHub 下载慢、或者 wheel 体积较大时。当前可用 wheel 约 `300MB`，直接在服务器端拉取通常更慢。

推荐流程：

1. 在本地下载与服务器环境匹配的 wheel。
2. 上传到 `Chen`，例如放到仓库根目录。
3. 在目标环境中执行本地 wheel 安装。

示例命令：

```bash
cd /data1/zsh/nlp_proj
uv pip install --no-deps ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

安装后可用下面的命令验证：

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
- `Exp A` 使用 RULER 下的 NIAH 子任务：`niah_single_1`、`niah_multikey_1`
- `Exp B` 使用 RULER reasoning 子任务和 LongBench 子集
- `Exp C` 复用 `Exp A / Exp B` 的数据
- `Exp D` 关注 TTFT、TPOT、throughput、peak GPU memory

模型主线：

- `Meta/Llama-3.1-8B-Instruct`
- `tiiuae/Falcon3-Mamba-7B-Instruct`

详见 [实验设计文档](docs/experiments.md)。