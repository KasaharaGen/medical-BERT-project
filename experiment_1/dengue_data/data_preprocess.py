import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

# 初回のみ必要
nltk.download('punkt')

# 数値・記号等の正規化関数（事前学習用）
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)                     # HTMLタグ除去
    text = re.sub(r'\[.*?\]', '', text)                   # 角カッコ内除去
    text = re.sub(r'\d+(\.\d+)?%', ' percent', text)      # パーセント → percent
    text = re.sub(r'\d+(\.\d+)?', ' num ', text)          # 数値 → num
    text = re.sub(r'[^a-z0-9\s\-/]', '', text)            # 記号除去（英数・空白・- / のみ残す）
    text = re.sub(r'\s+', ' ', text)                      # 空白正規化
    return text.strip()

# CSVファイルの読み込み（ファイルパスを適宜修正）
df = pd.read_csv("dengue_merged.csv")

# 文単位に分割し、新しい行として展開
split_rows = []

for idx, row in df.iterrows():
    abstract = str(row['Abstract'])  # NaN対策で文字列変換
    sentences = sent_tokenize(abstract)
    for sent in sentences:
        normalized = clean_text(sent)
        if len(normalized.split()) >= 5:  # 5語未満の短文は除外（任意）
            split_rows.append({
                'sentence': normalized
            })


df_split = pd.DataFrame(split_rows)
df_split.to_csv("dengue_sentences.csv", index=False)
print("✅ 正規化済み文データ saved to infectious_sentences.csv")