#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="fine_tuning_distill_optuna_ddp.py"

STUDENT_MODEL_DIR="../pretraining_bert_2/pretraining_bert_best/best_model"
TEACHER_MODEL_DIR="../pretraining_bert_2/pretraining_bert_best/best_model"
TOKENIZER_DIR="../pretraining_bert_2/pretraining_bert_best/tokenizer"
CSV_PATH="../data/learning_data.csv"

OUT_DIR="./result_distill_single_run"
SEED=42
MAX_LENGTH=512

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run_single_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] start: $(date)"

"${PYTHON_BIN}" -u "${SCRIPT}" \
  --student_model_dir "${STUDENT_MODEL_DIR}" \
  --teacher_model_dir "${TEACHER_MODEL_DIR}" \
  --tokenizer_dir     "${TOKENIZER_DIR}" \
  --csv               "${CSV_PATH}" \
  --output_dir        "${OUT_DIR}" \
  --seed              "${SEED}" \
  --max_length        "${MAX_LENGTH}" \
  2>&1 | tee "${LOG}"

echo "[INFO] done: $(date)"
