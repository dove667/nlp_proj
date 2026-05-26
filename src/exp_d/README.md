# Exp D: Serving System

目标：比较真实 serving 场景中的 TTFT、TPOT、throughput、peak GPU memory。

需要在服务器侧补齐：

- `client_adapter.py`：HF / vLLM / SGLang / 官方模型 client。
- `workload.py`：构造指定 context length、batch size、output length 的请求。
- `profiler.py`：采集 GPU memory、tokens/s、TTFT、TPOT。

入口：

```bash
conda run -n AI python scripts/run_exp_d.py --config configs/exp_d_serving.yaml
```
