# Exp B: Reasoning

`Exp B` 直接运行 [runner.py](/Users/dove/Desktop/NLP/nlp_proj/src/exp_b/runner.py)。

任务：

- `RULER` reasoning：`variable_tracking`、`common_words_aggregation`、`multi_hop_tracing`
- `LongBench` 子集：`hotpotqa`、`qasper`、`gov_report`、`repobench_p`

常用参数：

- `--model_name`
- `--architecture`
- `--implementation`
- `--model_path`
- `--tokenizer_path`
- `--output_dir`
- `--context_lengths`
- `--ruler_data_root` + `--ruler_tasks`
- `--longbench_data_root` + `--longbench_tasks`

查看完整参数：

```bash
python src/exp_b/runner.py --help
```
