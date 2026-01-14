#!/usr/bin/env bash
set -euo pipefail

# ========= ユーザー環境に合わせて編集 =========
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_DIR="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/fine_tuning"
SCRIPT="${PROJECT_DIR}/fine_tuning.py"

# student（pretraining_bert_2 の出力）
STUDENT_MODEL_DIR="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/pretraining_bert_2/pretraining_bert_best/best_model"

# teacher（強いモデルを推奨：別アーキテクチャも可）
TEACHER_BASE_DIR="${STUDENT_MODEL_DIR}"

# --- tokenizer を分離 ---
# student tokenizer
STUDENT_TOKENIZER_DIR="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/pretraining_bert_2/pretraining_bert_best/tokenizer"

# teacher tokenizer
# 1) teacherがstudentと同じなら、ここも同じでOK
# 2) DeBERTa等に差し替えるなら、そのteacherのtokenizerディレクトリ or HF名をローカルに落としたパスを指定
TEACHER_TOKENIZER_DIR="${TEACHER_BASE_DIR}"   # teacher_base_dirから読むならこれでOK（空でもOKだが明示推奨）

# 学習データ
CSV_PATH="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/data/learning_data.csv"

# 出力
OUT_DIR="${PROJECT_DIR}/result_distill_kfold_2"
STUDY_DIR="${PROJECT_DIR}/optuna_study"

# ========= Optuna設定 =========
# SQLite推奨（ENOSPC回避のため、十分空きがある場所へ）
OPTUNA_DB="/home/gonken2020/gen/medical-BERT-project/optuna/optuna_2.db"
STUDY_NAME="kd_seqcls_kfold"

N_TRIALS=30
N_STARTUP_TRIALS=8
USE_PRUNER=1   # 1=有効, 0=無効

# ========= 実行 =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

mkdir -p "${OUT_DIR}" "${STUDY_DIR}"
mkdir -p "$(dirname "${OPTUNA_DB}")"

PRUNER_FLAG=()
if [[ "${USE_PRUNER}" -eq 1 ]]; then
  PRUNER_FLAG+=(--use_pruner)
fi

"${PYTHON_BIN}" -u "${SCRIPT}" \
  --student_model_dir "${STUDENT_MODEL_DIR}" \
  --teacher_base_dir "${TEACHER_BASE_DIR}" \
  --student_tokenizer_dir "${STUDENT_TOKENIZER_DIR}" \
  --teacher_tokenizer_dir "${TEACHER_TOKENIZER_DIR}" \
  --csv "${CSV_PATH}" \
  --output_dir "${OUT_DIR}" \
  --study_dir "${STUDY_DIR}" \
  --test_ratio 0.2 \
  --n_splits 5 \
  --epochs 3 \
  --teacher_epochs 3 \
  --max_length 512 \
  --teacher_max_length 512 \
  --amp fp16 \
  --eval_steps 100 \
  --logging_steps 50 \
  --threshold_grid 401 \
  --use_optuna \
  --study_name "${STUDY_NAME}" \
  --n_trials "${N_TRIALS}" \
  --n_startup_trials "${N_STARTUP_TRIALS}" \
  --optuna_storage "sqlite:///${OPTUNA_DB}" \
  "${PRUNER_FLAG[@]}" \
  --final_train_best

echo "===================================================="
echo "Optuna artifacts:"
echo "  ${STUDY_DIR}/best_params.json"
echo "  ${STUDY_DIR}/trials_log.csv"
echo ""
echo "Final artifacts:"
echo "  ${OUT_DIR}/oof_result.json"
echo "  ${OUT_DIR}/chosen_threshold.json"
echo "  ${OUT_DIR}/final/curve_loss.png"
echo "  ${OUT_DIR}/final/curve_eval_metrics.png"
echo "  ${OUT_DIR}/final/test_confusion_matrix_percent_blues.png"
echo "===================================================="
