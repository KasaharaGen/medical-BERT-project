import sentencepiece as spm
import pandas as pd
import os
import shutil
from transformers import PreTrainedTokenizerFast

INPUT_CSV = "../data/merged_blocks.csv"
TEXT_COLUMN = "sentence"
CORPUS_PATH = "tokenizer_corpus.txt"
MODEL_PREFIX = "mybert_tokenizer"
MODEL_DIR = "./tokenizer"

os.makedirs(MODEL_DIR, exist_ok=True)

# === CSVからテキスト抽出・整形 ===
df = pd.read_csv(INPUT_CSV)
df = df[df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].str.strip() != "")]
with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for line in df[TEXT_COLUMN]:
        f.write(line.strip().replace("\n", " ") + "\n")

print(f"📄 コーパス書き出し完了: {CORPUS_PATH}")

# === 語彙サイズを動的に調整（最大14981に制限） ===
with open(CORPUS_PATH, encoding="utf-8") as f:
    num_lines = sum(1 for _ in f)
VOCAB_SIZE = min(14981, int(num_lines * 2))
print(f"🔤 語彙サイズを設定: VOCAB_SIZE = {VOCAB_SIZE}")

# === SentencePiece Unigram Tokenizer学習 ===
spm.SentencePieceTrainer.train(
    input=CORPUS_PATH,
    model_prefix=os.path.join(MODEL_DIR, MODEL_PREFIX),
    vocab_size=VOCAB_SIZE,
    character_coverage=0.9995,
    model_type="unigram",
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3,
    user_defined_symbols=["[MASK]"]
)

# === .model を tokenizer.model にリネームコピー ===
shutil.copyfile(f"{MODEL_DIR}/{MODEL_PREFIX}.model", f"{MODEL_DIR}/tokenizer.model")

# === Transformers形式で保存 ===
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    MODEL_DIR,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="[PAD]",
    mask_token="[MASK]"
)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Transformers形式で保存完了: {MODEL_DIR}/tokenizer_config.json")
