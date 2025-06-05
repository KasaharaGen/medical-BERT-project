import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

# === 設定 ===
INPUT_FILE = "../data/corpus.csv"  
OUTPUT_DIR = "tokenizer"
VOCAB_SIZE = 30000

# === 特殊トークンの定義（BERT標準）===
special_tokens = {
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
}

# === 入力チェック ===
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
    wordpieces_prefix="##",
    special_tokens=list(special_tokens.values())
)

# === Tokenizers形式で保存（vocab.txt） ===
tokenizer.save_model(OUTPUT_DIR)
print(f"[INFO] Saved WordPiece vocab to: {OUTPUT_DIR}")

# === Transformers互換形式で保存 ===
print("[INFO] Converting to HuggingFace compatible tokenizer...")
hf_tokenizer = BertTokenizerFast.from_pretrained(OUTPUT_DIR)

# === 特殊トークンをHuggingFace Tokenizerに登録 ===
hf_tokenizer.add_special_tokens(special_tokens)

# === 保存（tokenizer_config.json, special_tokens_map.jsonなどを含む） ===
hf_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] HuggingFace tokenizer saved to: {OUTPUT_DIR}")
