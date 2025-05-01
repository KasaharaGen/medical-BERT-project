import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os

# === 設定 ===
INPUT_CSV = "../data/ner_context_blocks.csv"
MODEL_NAME = "bert-base-uncased"  # 自作BERTに切り替え可
OUTPUT_DIR = "./checkpoints/bert_mlm"
BLOCK_COLUMN = "block"
MAX_LENGTH = 128
MLM_PROB = 0.15
NUM_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 8
USE_MULTIGPU = torch.cuda.device_count() > 1

# === データ読み込み ===
df = pd.read_csv(INPUT_CSV)
df = df[df[BLOCK_COLUMN].notna() & (df[BLOCK_COLUMN].str.strip() != "")]
dataset = Dataset.from_pandas(df[[BLOCK_COLUMN]].rename(columns={BLOCK_COLUMN: "text"}))

# === トークナイズ ===
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# === データコラレータ ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)

# === モデル準備 ===
model = BertForMaskedLM.from_pretrained(MODEL_NAME)
if USE_MULTIGPU:
    model = torch.nn.DataParallel(model)

# === 学習設定 ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True if torch.cuda.is_available() else False,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    report_to="none"
)

# === Trainer実行 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"🚀 学習開始 (GPU: {torch.cuda.device_count()}枚)")
trainer.train()
print("✅ MLM学習完了")
trainer.save_model(OUTPUT_DIR)
print(f"💾 モデル保存完了: {OUTPUT_DIR}")
