import json
import torch
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset

# === 設定 ===
CORPUS_PATH = "../data/merged_blocks_50%.csv"  # 学習用コーパス
TOKENIZER_PATH = "tokenizer"
BEST_PARAM_PATH = "output/bert_pretrain_optuna/best_params.json"  # ベストハイパーパラメータ
OUTPUT_DIR = "output/pretrained_bert"  # モデル保存先

# === トークナイザーとデータ ===
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

dataset = load_dataset("text", data_files={"train": CORPUS_PATH})
tokenized = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# === ベストハイパーパラメータ読み込み ===
with open(BEST_PARAM_PATH, "r") as f:
    best_params = json.loads(f.read().replace("'", '"'))  # dict形式に変換

# === モデル設定（BERT-base相当）===
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    pad_token_id=tokenizer.pad_token_id
)

model = BertForMaskedLM(config=config)

# === 学習設定 ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=32,
    num_train_epochs=5,
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    warmup_steps=500,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="tensorboard",  # ログ可視化
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === 学習実行 ===
trainer.train()

# === モデル保存 ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
