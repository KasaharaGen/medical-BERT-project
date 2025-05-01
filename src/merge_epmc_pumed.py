import pandas as pd

# ファイルパス
epmc_path = "../data/epmc_fulltext.csv"
pubmed_path = "../data/pubmed_abstracts.csv"
output_path = "../data/merged_fulltext.csv"

# データ読み込み
epmc_df = pd.read_csv(epmc_path)
pubmed_df = pd.read_csv(pubmed_path)

# EPMCとPubMedで共通のカラム名に整形
pubmed_df.rename(columns={
    "abstract": "fulltext",
    "pmid": "id"
}, inplace=True)

epmc_df["source"] = "epmc"
pubmed_df["source"] = "pubmed"

# カラム統一（必要に応じて調整）
common_cols = list(set(epmc_df.columns) & set(pubmed_df.columns))
merged_df = pd.concat([epmc_df[common_cols], pubmed_df[common_cols]], ignore_index=True)

# fulltext が空の行を除外
merged_df.dropna(subset=["fulltext"], inplace=True)
merged_df = merged_df[merged_df["fulltext"].str.strip() != ""]

# 重複除去（タイトルまたはIDなどで）
if "title" in merged_df.columns:
    merged_df.drop_duplicates(subset="title", inplace=True)

# 保存
merged_df.to_csv(output_path, index=False)
print(f"✅ 結合完了: {output_path}")