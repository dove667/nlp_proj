# NLP Course Project

本仓库是南方科技大学 NLP 课程项目仓库，主题是：**What Determines Long-Context Ability in LLMs? A Survey with Experimental Analysis**。

## 目录结构

- `docs/survey1.md`, `docs/survey2.md`, `docs/survey3.md`：文献综述与阅读笔记
- `docs/experiments.md`：实验主线与口径定义
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

`requirements.txt` 不包含 `torch`。建议先安装与服务器 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 实验口径

- 数据来源：`RULER + LongBench`
- `Exp A` 使用 RULER 下的 NIAH 子任务：`niah_single_1`、`niah_multikey_1`
- `Exp B` 使用 RULER reasoning 子任务和 LongBench 子集
- `Exp C` 复用 `Exp A / Exp B` 的数据
- `Exp D` 关注 TTFT、TPOT、throughput、peak GPU memory

模型主线：

- `Meta/Llama-3.1-8B-Instruct`
- `tiiuae/Falcon3-Mamba-7B-Instruct`
- `Zyphra/Zamba2-7B-Instruct-v2`

## 运行方式

### Exp A

```bash
python src/exp_a/eval_ruler_hf.py --help
```

### Exp B

```bash
python src/exp_b/runner.py --help
```

### Exp C

```bash
python src/exp_c/runner.py --help
```

### Exp D

```bash
python src/exp_d/runner.py --help
```
