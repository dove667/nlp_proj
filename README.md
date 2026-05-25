# NLP Course Project

欢迎来到本自然语言处理（NLP）课程项目的代码仓库。本项目包含了相关的文献调研（Surveys）以及实验记录（Experiments）。

## 目录结构 / Project Structure

- `survey1.md`, `survey2.md`, `survey3.md`: 课题相关的文献调研记录与阅读笔记。
- `experiments.md`: 核心实验步骤、参数设计以及实验结果的记录。
- *(其他代码、数据和模型目录可在此后补充)*

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

*(在此处补充如何运行你的实验代码，如数据预处理、模型训练、评测等)*

```bash
# python train.py
```
