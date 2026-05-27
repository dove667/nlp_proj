#!/usr/bin/env bash
set -euo pipefail

cd /data1/zsh/RULER/scripts/data

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NLTK_DATA=/data1/zsh/nltk_data

MODEL_NAME=llama31_8b_instruct # 替换成你的模型
TOKENIZER_PATH=/data1/zsh/models/Llama-3.1-8B-Instruct # 替换成你的模型
SAVE_ROOT=/data1/zsh/datasets/ruler
NUM_SAMPLES=500
TEMPLATE_TYPE=base

TASKS=(
  niah_single_1
  niah_multikey_1
)

LENGTHS=(
  4096
  8192
  16384
  32768
  65536
  131072
)

for max_len in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    save_dir="${SAVE_ROOT}/${MODEL_NAME}/${max_len}"

    echo "============================================================"
    echo "Generating RULER retrieval dataset"
    echo "model=${MODEL_NAME}"
    echo "task=${task}"
    echo "max_seq_length=${max_len}"
    echo "template=${TEMPLATE_TYPE}"
    echo "save_dir=${save_dir}"
    echo "============================================================"

    python prepare.py \
      --save_dir "${save_dir}" \
      --benchmark synthetic \
      --task "${task}" \
      --subset test \
      --tokenizer_type hf \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --max_seq_length "${max_len}" \
      --model_template_type "${TEMPLATE_TYPE}" \
      --num_samples "${NUM_SAMPLES}"
  done
done