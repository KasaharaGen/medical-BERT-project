import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
import os

# === パラメータ ===
csv_path = "../data/corpus.csv"
mesh_path = "mesh/mesh_dengue_infection_terms.txt"  # MeSH語リスト（1行1語）
vocab_size = 20000
priority_vocab_size = 10000  # TF-IDF上位語を優先

# === 正規化関数（事前学習用） ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\d+(\.\d+)?%', ' percent', text)
    text = re.sub(r'\d+(\.\d+)?', ' num ', text)
    text = re.sub(r'[^a-z0-9\s\-/]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === データ読み込み・前処理 ===
df = pd.read_csv(csv_path)
sentences = df["sentence"].dropna().astype(str).map(clean_text).tolist()

# === TF-IDF上位語抽出 ===
vectorizer = TfidfVectorizer(max_features=vocab_size * 2, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(sentences)
tfidf_top_words = vectorizer.get_feature_names_out()[:priority_vocab_size]

# === MeSH語リストとの照合 ===
with open(mesh_path) as f:
    mesh_terms = set(line.strip().lower() for line in f)

medical_terms = [word for word in tfidf_top_words if word in mesh_terms]
priority_vocab = list(dict.fromkeys(medical_terms + list(tfidf_top_words)))  # 重複排除し順序維持

# === トークナイザー学習用テキスト生成 ===
tmp_file = "tokenizer/abstracts_for_tokenizer.txt"
with open(tmp_file, "w", encoding="utf-8") as f:
    for sent in sentences:
        f.write(sent + "\n")

# === トークナイザー初期化（BPE） ===
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 正規化・前処理
tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])
tokenizer.pre_tokenizer = Whitespace()

# 特殊トークン
tokenizer_special = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# Trainer に seed vocabulary（優先語彙）を渡す
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=tokenizer_special,
    initial_alphabet=[],
    show_progress=True
)

# トークナイザー学習
tokenizer.train([tmp_file], trainer)

# Post-processor（BERT形式）
tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)

# 保存
tokenizer.save("tokenizer/tokenizer.json")
print("✅ Tokenizer trained and saved to tokenizer.json")
os.remove(tmp_file)
