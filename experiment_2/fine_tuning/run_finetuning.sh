#!/usr/bin/env bash
set -euo pipefail

# ==== 設定 ====
PY=python
SCRIPT=fine_tuning.py                              # あなたのPython本体
RESULT_DIR=./result
STUDY_DIR=${RESULT_DIR}/optuna_study
STUDY_NAME=bert_bin_tuning_mcc
DB_URI=sqlite:///${STUDY_DIR}/${STUDY_NAME}.db

# 総試行数（未指定なら100）
TOTAL_TRIALS="${TOTAL_TRIALS:-100}"

# GPU枚数の自動検出（失敗時は5）
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU="$(nvidia-smi --list-gpus | wc -l | xargs)"
else
  NGPU=5
fi
# 必要なら NGPU_OVERRIDE=5 などで上書き
NGPU="${NGPU_OVERRIDE:-$NGPU}"

echo "[INFO] TOTAL_TRIALS=${TOTAL_TRIALS}"
echo "[INFO] NGPU=${NGPU}"

# 各ワーカーが担当する試行数（切り上げ）
TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + NGPU - 1) / NGPU ))
echo "[INFO] TRIALS_PER_WORKER=${TRIALS_PER_WORKER}"

mkdir -p "${STUDY_DIR}"

# ==== 1) Optuna探索をGPU並列で実行 ====
echo "[INFO] Start parallel Optuna workers..."
PIDS=()
for (( i=0; i<NGPU; i++ )); do
  export CUDA_VISIBLE_DEVICES="${i}"
  # 最後の1本をマスター（探索完了後にbest_params.json保存 & 本学習トリガ役）
  if [[ "${i}" -eq "$((NGPU-1))" ]]; then
    OPTUNA_MASTER=1
  else
    OPTUNA_MASTER=0
  fi

  echo "[INFO] launch worker on GPU=${i}, OPTUNA_MASTER=${OPTUNA_MASTER}"

  STORAGE="${DB_URI}" \
  TRIALS_PER_WORKER="${TRIALS_PER_WORKER}" \
  ENABLE_TUNING=1 \
  OPTUNA_MASTER="${OPTUNA_MASTER}" \
  TOKENIZERS_PARALLELISM=false \
  ${PY} "${SCRIPT}" &
  PIDS+=($!)
done

# 全ワーカー終了待ち
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done
echo "[INFO] All Optuna workers finished."

# ==== 2) ベストパラメータ確認 ====
BEST_JSON="${RESULT_DIR}/best_params.json"
if [[ ! -f "${BEST_JSON}" ]]; then
  echo "[ERROR] ${BEST_JSON} が見つからない。マスター側が正常終了したか確認せよ。"
  exit 1
fi
echo "[INFO] Best params file: ${BEST_JSON}"
cat "${BEST_JSON}"

# ==== 3) 最良パラメータでDDP本学習（torchrun） ====
# 重要: ENABLE_TUNING=False を付与して探索を無効化し、再学習のみ実行する
echo "[INFO] Start final training with torchrun (DDP) using best params..."
ENABLE_TUNING=False TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node="${NGPU}" \
  --master_port=29571 \
  "${SCRIPT}" \
  2>&1 | tee "${RESULT_DIR}/final_ddp_train.log"

echo "[INFO] Finished. Check artifacts under: ${RESULT_DIR}"
echo "[INFO] - best_params.json"
echo "[INFO] - test_metrics.json / test_metrics.csv"
echo "[INFO] - test_confusion_matrix.csv / test_confusion_matrix.png"
echo "[INFO] - curve_loss.png / curve_eval_metrics.png"
