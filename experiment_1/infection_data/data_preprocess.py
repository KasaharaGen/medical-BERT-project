import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

# 初回のみ必要
nltk.download('punkt')

# === 正規化関数（事前学習用）===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)                     # HTMLタグ除去
    text = re.sub(r'\[.*?\]', '', text)                   # 角カッコ除去
    text = re.sub(r'\d+(\.\d+)?%', ' percent', text)      # パーセント → percent
    text = re.sub(r'\d+(\.\d+)?', ' num ', text)          # 数値 → num
    text = re.sub(r'[^a-z0-9\s\-/]', '', text)            # 記号除去（- / は許容）
    text = re.sub(r'\s+', ' ', text)                      # 空白正規化
    return text.strip()

# === CSV読み込み ===
df = pd.read_csv("infectious_disease_merged.csv")

# 文分割 + 正規化
split_rows = []
for idx, row in df.iterrows():
    abstract = str(row['Abstract'])  # NaN対策で文字列化
    sentences = sent_tokenize(abstract)
    for sent in sentences:
        cleaned = clean_text(sent)
        if len(cleaned.split()) >= 5:  # 5語以上の文のみ使用
            split_rows.append({
                'sentence': cleaned
            })

# 出力
df_split = pd.DataFrame(split_rows)
df_split.to_csv("infectious_sentences.csv", index=False)
print("✅ 正規化済み文データ saved to infectious_sentences.csv")
