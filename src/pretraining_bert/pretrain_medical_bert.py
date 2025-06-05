import json
import torch
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import os

# === 設定 ===
CORPUS_PATH = "../../data/corpus.csv"
TOKENIZER_PATH = "tokenizer"
BEST_PARAM_PATH = "output/bert_pretrain_optuna/best_params.json"
OUTPUT_DIR = "output/pretrained_bert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === トークナイザー ===
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)

# === データ読み込み・分割 ===
dataset_all = load_dataset("text", data_files={"data": CORPUS_PATH})["data"]
split = dataset_all.train_test_split(test_size=0.05, seed=42)
train_dataset_raw = split["train"]
eval_dataset_raw = split["test"]

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

train_dataset = train_dataset_raw.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset_raw.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.2
)

# === ベストハイパーパラメータ読み込み ===
with open(BEST_PARAM_PATH, "r") as f:
    best_params = json.loads(f.read().replace("'", '"'))

# === モデル設定 ===
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

# === 学習設定（EarlyStoppingとベストモデル復元含む） ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,                     # VRAMに応じて調整
    num_train_epochs=10,                                 # patienceで早期終了
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    warmup_steps=500,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,                         # 👈 ベストモデルを使用
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 👈 EarlyStopping有効化
)

# === 学習実行 ===
trainer.train()

# === モデル保存 ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === ログ保存・学習曲線プロット ===
log_history = trainer.state.log_history
log_df = pd.DataFrame(log_history)

train_df = log_df[log_df["loss"].notnull()][["step", "loss"]]
eval_df = log_df[log_df["eval_loss"].notnull()][["step", "eval_loss"]]

# 保存パス
loss_csv = os.path.join(OUTPUT_DIR, "training_loss.csv")
eval_csv = os.path.join(OUTPUT_DIR, "eval_loss.csv")
loss_plot = os.path.join(OUTPUT_DIR, "loss_curve.png")

# CSV出力
train_df.to_csv(loss_csv, index=False)
eval_df.to_csv(eval_csv, index=False)

# プロット
plt.figure(figsize=(8, 5))
plt.plot(train_df["step"], train_df["loss"], label="Training Loss")
plt.plot(eval_df["step"], eval_df["eval_loss"], label="Eval Loss", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("BERT Pretraining Loss (Train vs Eval)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(loss_plot)
plt.show()

print(f"✅ Train loss log saved to: {loss_csv}")
print(f"✅ Eval loss log saved to:  {eval_csv}")
print(f"✅ Loss curve plot saved to: {loss_plot}")
