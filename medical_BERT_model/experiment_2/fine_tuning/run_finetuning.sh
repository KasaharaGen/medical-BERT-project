#!/usr/bin/env bash
set -euo pipefail

# ==== 設定（必要に応じて変更）====
SCRIPT=fine_tuning.py
GPUS=(0 1 2 3 4)          # 使用GPU
TOTAL_TRIALS=2000          # 総試行数（例）
DB_DIR=./result/optuna_study
DB_PATH=${DB_DIR}/bert_bin_tuning_mcc.db

mkdir -p "${DB_DIR}"

NUM_WORKERS=${#GPUS[@]}
BASE_TRIALS=$(( TOTAL_TRIALS / NUM_WORKERS ))
REM=$(( TOTAL_TRIALS % NUM_WORKERS ))

echo "==== Phase 0: 事前チェック ===="
# スクリプト側のENABLE_TUNING=True を前提にする:contentReference[oaicite:1]{index=1}
grep -q 'ENABLE_TUNING = True' "${SCRIPT}" || {
  echo "[ERROR] ${SCRIPT} 内の ENABLE_TUNING を True にしてから実行すること。" >&2
  exit 1
}
echo "DB: ${DB_PATH}"
echo "総試行数: ${TOTAL_TRIALS} / ワーカー数: ${NUM_WORKERS}"
echo "各ワーカー試行数: 基本 ${BASE_TRIALS} + 先頭 ${REM} ワーカーに +1"

echo
echo "==== Phase 1: 探索のみ（5GPU 並列）===="
pids=()
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  trials=${BASE_TRIALS}
  if (( i < REM )); then
    trials=$(( trials + 1 ))
  fi

  echo "[GPU${gpu}] Worker 起動: TRIALS_PER_WORKER=${trials}"
  # Phase1は全員 Worker 扱い（最終学習はしない）
  CUDA_VISIBLE_DEVICES=${gpu} \
  OPTUNA_MASTER=0 \
  TRIALS_PER_WORKER=${trials} \
  python "${SCRIPT}" &

  pids+=($!)
done

# すべての探索Worker終了を待つ
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "==== Phase 1 完了：全ワーカー探索終了 ===="

echo
echo "==== Phase 2: master が最良ハイパラで最終学習 ===="
# Phase2は探索を追加実行せず（0試行）、DBからbestを読み出して学習のみ:contentReference[oaicite:2]{index=2}
CUDA_VISIBLE_DEVICES=${GPUS[0]} \
OPTUNA_MASTER=1 \
TRIALS_PER_WORKER=0 \
python "${SCRIPT}"

echo "==== 全処理完了 ===="
