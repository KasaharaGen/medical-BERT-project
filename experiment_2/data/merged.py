import pandas as pd

# CSVファイルの読み込み
df1 = pd.read_csv("infection_data/infectious_sentences.csv")
df2 = pd.read_csv("dengue_data/dengue_sentences.csv")

# 縦方向に結合（行を追加）
df_combined = pd.concat([df1, df2], ignore_index=True)

# 'text'列を小文字に変換（列名は適宜変更）
df_combined['sentence'] = df_combined['sentence'].str.lower()

# 結果を保存
df_combined.to_csv("corpus.csv", index=False)
