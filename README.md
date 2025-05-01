#medical-BERT-project
# ディレクトリ構成:
# project/
# ├── pretrain/train_mlm.py
# ├── finetune/train_classifier.py
# ├── tokenizer/train_tokenizer.py
# ├── utils/dataset.py
# ├── deploy/app.py
# ├── data/epmc_fulltext.csv
# └── data/ehr_train.csv

# このプロジェクトは、論文のMLM事前学習と電子カルテの病気分類に分かれた次のステップで構成される:

# === 1. 論文MLM事前学習 ===
# train_mlm.py
# - SentencePiece tokenizerでの分かち込み
# - BERTEncoderのMLM学習

# === 2. 電子カルテ分類 ===
# train_classifier.py
# - BERTベースのファインチューニング
# - EHRデータを用いたラベル分類

# === 3. デプロイ ===
# app.py
# - FastAPI を使用した接続APIの実装

# === 4. トークナイザー ===
# train_tokenizer.py
# - SentencePiece でトークナイザの訓練

# === 5. データセット ===
# dataset.py
# - 事前学習用コーパス
# - 病気分類用Datasetの定義
