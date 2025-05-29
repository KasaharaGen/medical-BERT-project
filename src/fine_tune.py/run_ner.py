import subprocess
import glob
import os
import pandas as pd

# === 設定 ===
chunk_dir = "data/chunks"
output_dir = "data/processed"
GPU_IDS = [0, 1]
output_corpus_path = "data/learning_data.csv"

# === ディレクトリ存在チェック ===
if not os.path.isdir(chunk_dir):
    raise FileNotFoundError("[ERROR] chunks ディレクトリが存在しません")
if not os.path.isdir(output_dir):
    raise FileNotFoundError("[ERROR] processed ディレクトリが存在しません")

# === チャンクごとに処理実行 ===
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

print("✅ すべてのチャンク処理完了")

# === 出力の統合 ===
print("[STEP] NER結果を統合中...")
block_files = sorted(glob.glob(os.path.join(output_dir, "*_blocks.csv")))
all_blocks = []
for file in block_files:
    df = pd.read_csv(file)
    all_blocks.append(df)

if all_blocks:
    merged_df = pd.concat(all_blocks, ignore_index=True)
    merged_df.to_csv(output_corpus_path, index=False)
    print(f"✅ 統合完了: {len(merged_df)} 件を {output_corpus_path} に保存")
else:
    print("[WARN] 統合対象のブロックファイルが見つかりませんでした")
