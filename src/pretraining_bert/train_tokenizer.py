import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

# === 設定 ===
INPUT_FILE = "../data/corpus.csv"  
OUTPUT_DIR = "tokenizer"
VOCAB_SIZE = 30000

# === チェック ===
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === トークナイザー学習（WordPiece）===
print("[INFO] Training WordPiece tokenizer...")
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=True,
    lowercase=True
)

tokenizer.train(
    files=[INPUT_FILE],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

# === 保存（Tokenizers形式） ===
tokenizer.save_model(OUTPUT_DIR)
print(f"[INFO] Saved WordPiece vocab to: {OUTPUT_DIR}")

# === Transformers互換形式で保存 ===
print("[INFO] Converting to HuggingFace compatible tokenizer...")
hf_tokenizer = BertTokenizerFast.from_pretrained(OUTPUT_DIR)
hf_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] HuggingFace tokenizer saved to: {OUTPUT_DIR}")
