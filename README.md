# NLP Course Project

欢迎来到本自然语言处理（NLP）课程项目的代码仓库。本项目包含了相关的文献调研（Surveys）以及实验记录（Experiments）。

## 目录结构 / Project Structure

- `docs/survey1.md`, `docs/survey2.md`, `docs/survey3.md`: 课题相关的文献调研记录与阅读笔记。
- `docs/experiments.md`: 核心实验步骤、参数设计以及实验结果的记录。
- `configs/`: 实验配置文件。
- `scripts/`: 可直接运行的实验脚本与绘图脚本。
- `src/exp_a/`: Exp A，跨架构 retrieval 基线，覆盖 NIAH 和 RULER retrieval。
- `src/exp_b/`: Exp B，跨架构 reasoning 测试，覆盖 RULER 复杂任务和 LongBench 子集。
- `src/exp_c/`: Exp C，推理时扩窗/优化方法测试。
- `src/exp_d/`: Exp D，serving 系统吞吐、延迟和显存测试。
- `results/`: 实验输出目录，保存 JSONL、CSV、summary 和图表。

## 环境依赖 / Setup

建议使用 Anaconda 或虚拟环境来管理项目依赖：

```bash
conda create -n nlp_env python=3.11
conda activate nlp_env
```

由于不同设备的 CUDA 环境不同，本项目的 `requirements.txt` 中未包含 PyTorch。请在安装其他依赖前，先前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 找到适合你设备的安装命令。

例如，对于 CUDA 12.1 环境，可以使用：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装完 PyTorch 后，再安装其余依赖：
```bash
pip install -r requirements.txt
```

## 运行说明 / Usage

当前代码按 `docs/experiments.md` 的四个实验拆成四套框架。默认假设模型、数据集和第三方推理方法已经在服务器上准备好，本仓库只提供实验编排、配置、结果格式和适配器接口。

### Exp A：跨架构 Retrieval 基线

```bash
conda run -n AI python scripts/run_exp_a.py --config configs/exp_a_retrieval.yaml
```

需要补齐 `src/exp_a/data_loader.py` 和 `src/exp_a/model_adapter.py`，接入服务器上的 NIAH / RULER retrieval 数据和四类模型。

### Exp B：跨架构 Reasoning 测试

```bash
conda run -n AI python scripts/run_exp_b.py --config configs/exp_b_reasoning.yaml
```

需要补齐 `src/exp_b/data_loader.py`、`src/exp_b/model_adapter.py`，并按数据集格式完善 F1 / ROUGE-L scoring。

### Exp C：推理时优化方法测试

```bash
conda run -n AI python scripts/run_exp_c.py --config configs/exp_c_inference_extension.yaml
```

需要在 `src/exp_c/method_adapter.py` 中接入 baseline、YaRN、Self-Extend、KIVI、SnapKV、FIER、StreamingLLM 等方法。

### Exp D：系统性能测试

```bash
conda run -n AI python scripts/run_exp_d.py --config configs/exp_d_serving.yaml
```

需要在 `src/exp_d/client_adapter.py` 和 `src/exp_d/profiler.py` 中接入 HF / vLLM / SGLang client、GPU 显存采集和更精确的 TTFT / TPOT 记录。
