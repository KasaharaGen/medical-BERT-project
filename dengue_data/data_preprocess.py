import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# 初回のみ必要
nltk.download('punkt')
nltk.download('punkt_tab')


# CSVファイルの読み込み（ファイルパスを適宜修正）
df = pd.read_csv("dengue_merged.csv")

# 文単位に分割し、新しい行として展開
split_rows = []

for idx, row in df.iterrows():
    abstract = str(row['Abstract'])  # NaN対策で文字列変換
    sentences = sent_tokenize(abstract)
    for sent in sentences:
        split_rows.append({
        'sentence': sent.strip().lower()
        })

# 分割結果をデータフレームに変換
df_split = pd.DataFrame(split_rows)

# CSVとして出力
df_split.to_csv("dengue_sentences.csv", index=False)
