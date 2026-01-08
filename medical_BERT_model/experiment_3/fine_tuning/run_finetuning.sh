#!/usr/bin/env bash
set -euo pipefail

# =========================
# 設定（ここを自分の環境に合わせて変更）
# =========================
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT="fine_tuning_distill_optuna_ddp.py"

STUDENT_MODEL_DIR="../pretraining_bert_2/pretraining_bert_best/best_model"
TEACHER_MODEL_DIR="../pretraining_bert_2/pretraining_bert_best/best_model"
TOKENIZER_DIR="../pretraining_bert_2/pretraining_bert_best/tokenizer"
CSV_PATH="../data/learning_data.csv"

OUT_DIR="./result_distill_single"
STUDY_DIR="${OUT_DIR}/optuna_study"

N_TRIALS=30
SEED=42

# VRAMが厳しければ max_length を落とす（512→256など）
MAX_LENGTH=512

# =========================
# 実行
# =========================
mkdir -p "${OUT_DIR}" "${STUDY_DIR}"

LOG="${OUT_DIR}/run_optuna_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] start: $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] output_dir=${OUT_DIR}"
echo "[INFO] study_dir=${STUDY_DIR}"
echo "[INFO] log=${LOG}"

"${PYTHON_BIN}" -u "${SCRIPT}" \
  --student_model_dir "${STUDENT_MODEL_DIR}" \
  --teacher_model_dir "${TEACHER_MODEL_DIR}" \
  --tokenizer_dir     "${TOKENIZER_DIR}" \
  --csv               "${CSV_PATH}" \
  --output_dir        "${OUT_DIR}" \
  --use_optuna \
  --study_dir         "${STUDY_DIR}" \
  --n_trials          "${N_TRIALS}" \
  --seed              "${SEED}" \
  --max_length        "${MAX_LENGTH}" \
  --final_train_best \
  2>&1 | tee "${LOG}"

echo "[INFO] done: $(date)"
