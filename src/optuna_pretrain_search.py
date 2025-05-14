import optuna
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset
import torch
import os

# === 設定 ===
CORPUS_PATH = "../data/merged_blocks_50%.csv"  # 学習用コーパス
TOKENIZER_PATH = "tokenizer"
MODEL_OUTPUT_DIR = "output/bert_pretrain_optuna"

# === トークナイザー ===
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)

# === データ読み込み ===
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

dataset = load_dataset("text", data_files={"train": CORPUS_PATH})
tokenized = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["text"])

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# === 評価関数 ===
def objective(trial):
    # 探索空間
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    # モデル構成
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

    # Training設定
    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUTPUT_DIR}/trial_{trial.number}",
        per_device_train_batch_size=16,
        num_train_epochs=1,  # チューニング時は短く（後で本学習）
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        evaluation_strategy="no"  # eval loss取得用に工夫（Trainer内部で取得）
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer.state.log_history[-1]["loss"]

# === Optuna実行 ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# === 最良パラメータ表示 ===
print("Best trial:")
print(study.best_trial)

# 結果保存
with open(f"{MODEL_OUTPUT_DIR}/best_params.txt", "w") as f:
    f.write(str(study.best_params))
