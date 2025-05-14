import pandas as pd
import os
import re
from tqdm import tqdm
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# === 設定 ===
FILES = [
    "../data/medrxiv_fulltext.csv",
    "../data/biorxiv_fulltext.csv",
    "../data/epmc_fulltext.csv"
]
OUTPUT_PATH = "../data/merged_fulltext.csv"
SENTENCE_OUTPUT_PATH = "../data/merged_fulltext_sentences.csv"

# === データ統合 ===
all_dfs = []
for file in FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df["source"] = os.path.basename(file).split("_")[0]
        all_dfs.append(df)
    else:
        print(f"[WARN] ファイルが存在しません: {file}")

merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df = merged_df.drop_duplicates(subset=["title"], keep="first")
merged_df = merged_df[~merged_df["fulltext"].isna() & (merged_df["fulltext"].str.strip() != "")]
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ 統合保存完了: {OUTPUT_PATH}")

# === 文単位に分割 ===
sentence_records = []
print("fulltextを文単位に分割中...")
for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    text = row.get("fulltext", "")
    if not isinstance(text, str) or not text.strip():
        continue
    try:
        sentences = sent_tokenize(text)
        for sent in sentences:
            sentence_records.append({
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "sentence": sent.strip()
            })
    except Exception as e:
        print(f"[WARN] Sentence split failed: {e}")

sentence_df = pd.DataFrame(sentence_records)
sentence_df.to_csv(SENTENCE_OUTPUT_PATH, index=False)
print(f"✅ 文単位ファイル出力完了: {SENTENCE_OUTPUT_PATH}")