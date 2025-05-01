# 🧠 BERT-based Disease Classification from Medical Records (研究目的)

本プロジェクトは、**医療論文を用いたBERTモデルの事前学習**と、**電子カルテ（EHR）に基づく疾患分類モデルの構築**を目的としています。  
商用利用は想定せず、研究目的に限って開発・検証されています。

## 🔁 プロジェクト構成
project/ ├── data/ │ ├── epmc_fulltext.csv # 医療論文（事前学習用） │ └── ehr_train.csv # 電子カルテ（分類タスク用） ├── tokenizer/ │ └── train_tokenizer.py # SentencePieceトークナイザの学習 ├── pretrain/ │ └── train_mlm.py # 論文コーパスによるMLM学習（自作BERT） ├── finetune/ │ └── train_classifier.py # 疾患分類モデルのファインチューニング ├── utils/ │ └── dataset.py # Dataset定義（事前学習・分類用） └── deploy/ └── app.py # FastAPIによるAPIデプロイ

## 📚 モデル構築フロー

1. **SentencePieceトークナイザの学習**  
   `tokenizer/train_tokenizer.py` で独自の語彙辞書を学習します。

2. **医療論文コーパスでBERTを事前学習**  
   `pretrain/train_mlm.py` でMLM（Masked Language Modeling）により、自作BERTを訓練します。

3. **電子カルテによる疾患分類ファインチューニング**  
   `finetune/train_classifier.py` で医療テキストから疾患（例：デング熱）を分類します。

4. **FastAPIによるモデルAPIの提供**  
   `deploy/app.py` を用いてWebサービスとして推論APIを構築します。

## ⚠️ 免責事項

- このプロジェクトは研究目的にのみ使用されます。
- 医療診断を目的とした臨床使用や商用利用は行いません。
- データセット（epmc_fulltext.csv, ehr_train.csv）は著作権・利用制限に配慮して管理してください。

## 📩 開発・問い合わせ

研究・学術交流目的の利用に関心がある方はご連絡ください。

