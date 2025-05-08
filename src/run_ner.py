import subprocess
import glob
import os
from time import sleep
import pandas as pd

chunk_dir = "../data/chunks"
output_dir = "../data/processed"
GPU_IDS = [0, 1]

if not os.path.isdir(chunk_dir):
    raise FileNotFoundError("[ERROR] ../data/chunks ディレクトリが存在しません")
if not os.path.isdir(output_dir):
    raise FileNotFoundError("[ERROR] ../data/processed ディレクトリが存在しません")

chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.csv")))
processes = []

for i, path in enumerate(chunk_paths):
    gpu_id = GPU_IDS[i % len(GPU_IDS)]
    cmd = ["python", "process_worker.py", path, str(gpu_id)]
    print(f"[RUN] {path} on GPU {gpu_id}")
    proc = subprocess.Popen(cmd)
    processes.append(proc)

    if (i + 1) % len(GPU_IDS) == 0:
        for p in processes:
            p.wait()
        processes = []

for p in processes:
    p.wait()

print("すべてのチャンク処理完了")

# NER結果の統合処理
print("[STEP] NER結果を統合中...")
block_files = sorted(glob.glob(os.path.join(output_dir, "*_blocks.csv")))
all_blocks = []
for file in block_files:
    df = pd.read_csv(file)
    all_blocks.append(df)

if all_blocks:
    merged_blocks = pd.concat(all_blocks, ignore_index=True)
    merged_blocks.to_csv("../data/merged_blocks.csv", index=False)
    print(f"✅ 統合完了: {len(merged_blocks)} 件を ../data/merged_blocks_50%.csv に保存")
else:
    print("[WARN] 統合対象のブロックファイルが見つかりませんでした")