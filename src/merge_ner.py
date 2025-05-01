import pandas as pd
import os
import re
from tqdm import tqdm
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# === 設定 ===
FILES = [
    "../data/epmc_fulltext.csv",
    "../data/pubmed_abstracts.csv",
    "../data/medrxiv_fulltext.csv",
    "../data/biorxiv_fulltext.csv"
]
OUTPUT_PATH = "../data/merged_fulltext.csv"
EXTRACTED_PATH = "../data/ner_context_blocks.csv"
KEYWORDS = ["dengue", "covid19", "malaria", "sars coronavirus", "mars coronavirus"]
KEYWORDS = [kw.lower() for kw in KEYWORDS]
MODEL_NAME = "d4data/biobert-chemical-disease-ner"

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

# === NER モデル準備 ===
print("🧠 モデル読み込み中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# === NER + キーワード文脈抽出 ===
def extract_blocks_with_ner(text, keywords):
    sentences = sent_tokenize(text)
    blocks = []
    for i, sent in enumerate(sentences):
        ner_results = ner_pipeline(sent)
        entities = [e["word"].lower() for e in ner_results if e["entity_group"] in ["DISEASE", "CHEMICAL"]]
        if any(kw in entity for kw in keywords for entity in entities):
            start = max(i - 2, 0)
            end = min(i + 3, len(sentences))
            block = " ".join(sentences[start:end])
            blocks.append(block)
    return blocks

records = []
print("🔍 NERターゲット文を抽出中...")
for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    text = row.get("fulltext", "")
    if not isinstance(text, str) or not text.strip():
        continue
    blocks = extract_blocks_with_ner(text, KEYWORDS)
    for b in blocks:
        records.append({
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "block": b
        })

out_df = pd.DataFrame(records)
out_df.to_csv(EXTRACTED_PATH, index=False)
print(f"✅ NERブロック出力完了: {EXTRACTED_PATH}")
