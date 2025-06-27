import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
import os

# === CSVからsentenceを読み込む（前処理付き） ===
csv_path = "../data/corpus.csv"
df = pd.read_csv(csv_path)

# 文正規化関数（前処理強化）
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\d+(\.\d+)?%', ' percent', text)
    text = re.sub(r'\d+(\.\d+)?', ' num ', text)
    text = re.sub(r'[^a-z0-9\s\-/]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 文単位前処理
sentences = df["sentence"].dropna().astype(str).map(clean_text).tolist()

# === 医学語選定（TF-IDF） ===
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(sentences)
top_vocab = vectorizer.get_feature_names_out()

# 任意の医学語辞書と照合（省略可、ここでは全TF-IDF語を採用）
with open("tfidf_vocab.txt", "w") as f:
    for word in top_vocab:
        f.write(word + "\n")

# === テキストファイル出力（トークナイザー学習用） ===
tmp_file = "abstracts_for_tokenizer.txt"
with open(tmp_file, "w", encoding="utf-8") as f:
    for sent in sentences:
        f.write(sent + "\n")

# === トークナイザー初期化（BPE） ===
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 正規化
tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])

# 前処理（空白単位）
tokenizer.pre_tokenizer = Whitespace()

# 特殊トークン
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=20000, special_tokens=special_tokens)

# トークナイザー学習
tokenizer.train([tmp_file], trainer)

# Post-processor（BERT互換）
tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)

# 保存
tokenizer.save("tokenizer.json")
print("✅ Tokenizer trained and saved to tokenizer.json")

# 一時ファイル削除
os.remove(tmp_file)
