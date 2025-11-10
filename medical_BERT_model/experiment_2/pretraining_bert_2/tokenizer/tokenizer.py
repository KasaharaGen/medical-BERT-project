import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
import os

# === CSVからsentenceを読み込む ===
csv_path = "../../dengue_data/dengue_sentences.csv"  # CSVファイルパス
df = pd.read_csv(csv_path)
sentence = df["sentence"].dropna().astype(str).tolist()

# === 一時ファイルにテキストを書き出す ===
tmp_file = "abstracts_for_tokenizer.txt"
with open(tmp_file, "w", encoding="utf-8") as f:
    for abs_text in sentence:
        abs_text = abs_text.strip().replace("\n", " ")
        f.write(abs_text + "\n")

# === トークナイザー初期化 ===
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 正規化処理を追加（NFD → 小文字化 → アクセント除去）
tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])

# 空白で単語単位に分割
tokenizer.pre_tokenizer = Whitespace()

# 特殊トークンの定義
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=20000, special_tokens=special_tokens)

# 学習実行（ファイルはリストで指定）
tokenizer.train([tmp_file], trainer)

# Post-processor（BERT互換のCLS/SEPを自動付与）
tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)

# トークナイザー保存
tokenizer.save("tokenizer.json")
print("✅ Tokenizer trained and saved to tokenizer.json")

# 一時ファイル削除（任意）
os.remove(tmp_file)
