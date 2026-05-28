#!/usr/bin/env bash
set -euo pipefail

cd /data1/zsh/RULER/scripts/data

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NLTK_DATA=/data1/zsh/nltk_data

MODEL_NAME=Falcon3-Mamba-7B-Instruct
TOKENIZER_PATH=/data1/zsh/models/Falcon3-Mamba-7B-Instruct
SAVE_ROOT=/data1/zsh/datasets/ruler/reasoning
NUM_SAMPLES=500
TEMPLATE_TYPE=base

# RULER reasoning / aggregation subtasks
# vt  = variable tracking / multi-hop tracing
# cwe = common words extraction / aggregation
# fwe = frequent words extraction / aggregation
TASKS=(
  vt
  cwe
  fwe
)

LENGTHS=(
  4096
  8192
  16384
  32768
)

for max_len in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    save_dir="${SAVE_ROOT}/${MODEL_NAME}/${max_len}"

    echo "============================================================"
    echo "Generating RULER reasoning/aggregation dataset"
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