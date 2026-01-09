#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="gen"
PROJECT_ROOT="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/fine_tuning"
PYTHON_BIN="$HOME/anaconda3/envs/${ENV_NAME}/bin/python"

STUDENT_MODEL_DIR="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/pretraining_bert_2/pretraining_bert_best/best_model"
TEACHER_BASE_DIR="${STUDENT_MODEL_DIR}"
TOKENIZER_DIR="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/pretraining_bert_2/pretraining_bert_best/tokenizer"

CSV_PATH="/home/gonken2020/gen/medical-BERT-project/medical_BERT_model/experiment_3/data/learning_data.csv"

OUT_DIR="${PROJECT_ROOT}/result_kfold_kd_lora_single"
STUDY_DIR="${OUT_DIR}/optuna_study"  # 使わないが引数上必要なので作る

mkdir -p "${OUT_DIR}" "${STUDY_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" fine_tuning.py \
  --student_model_dir "${STUDENT_MODEL_DIR}" \
  --teacher_base_dir  "${TEACHER_BASE_DIR}" \
  --tokenizer_dir     "${TOKENIZER_DIR}" \
  --csv               "${CSV_PATH}" \
  --output_dir        "${OUT_DIR}" \
  --study_dir         "${STUDY_DIR}" \
  --use_kfold \
  --n_splits 5 \
  --test_ratio 0.2 \
  --final_train_best \
  --max_length 512 \
  --batch_size 8 \
  --grad_accum 4 \
  --epochs 3 \
  --teacher_epochs 2 \
  --distill_alpha 0.35 \
  --temperature 2.0 \
  --rep_beta 0.5 \
  --prior_tau 1.0 \
  --class_weight_power 0.5 \
  --class_weight_clip 3.0 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --label_smoothing 0.05 \
  --amp fp16 \
  --cm_percent_mode all \
  --use_lora \
  --lora_targets "query,key,value,dense" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05

echo "[INFO] Done. Outputs:"
echo "  - Final test CM (Blues%): ${OUT_DIR}/final/test_confusion_matrix_percent_blues.png"
echo "  - Test metrics (fixed thr): ${OUT_DIR}/final/test_metrics_fixed_threshold.json"
