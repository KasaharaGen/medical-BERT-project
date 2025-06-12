import os
import sys
import optuna
import torch
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset

# === GPU指定 ===
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# === 設定 ===
CORPUS_PATH = "../../data/corpus.csv"
TOKENIZER_PATH = "tokenizer"
MODEL_OUTPUT_DIR = "output/bert_pretrain_optuna"
DB_PATH = "sqlite:///optuna.db"
STUDY_NAME = "bert_pretrain_gpu"

# === トークナイザーとデータ ===
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
dataset = load_dataset("text", data_files={"train": CORPUS_PATH})

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

# 学習/検証分割
split = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
eval_dataset = split["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

# マスク化処理
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.2
)

# === Optuna評価関数 ===
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.1)

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

    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUTPUT_DIR}/trial_{trial.number}",
        per_device_train_batch_size=64,          # VRAMに応じて調整
        num_train_epochs=10,                      # EarlyStoppingを前提に多めに
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]

# === Optuna Studyロード & 最適化実行 ===
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
study.optimize(objective, n_trials=5)
