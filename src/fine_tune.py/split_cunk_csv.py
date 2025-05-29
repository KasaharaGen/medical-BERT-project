import os
import pandas as pd
import math

INPUT_PATH = "data/corpus.csv"
OUTPUT_DIR = "data/chunks"
CHUNK_SIZE = 5000  # 1チャンクあたり

def split_csv_by_chunk(input_path, output_dir, chunk_size=5000):
    if not os.path.isfile(input_path):
        raise FileNotFoundError("[ERROR] 入力ファイルが存在しません: " + input_path)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError("[ERROR] チャンク出力ディレクトリが存在しません: " + output_dir)

    df = pd.read_csv(input_path)
    total_chunks = math.ceil(len(df) / chunk_size)

    for i in range(total_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(output_dir, f"part_{i:03d}.csv")
        chunk_df.to_csv(chunk_path, index=False)
        print(f"✅ 出力: {chunk_path} ({len(chunk_df)} 行)")

    print(f"全{total_chunks}チャンクに分割完了")

if __name__ == "__main__":
    split_csv_by_chunk(INPUT_PATH, OUTPUT_DIR, CHUNK_SIZE)
