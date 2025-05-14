import os
import sys
import optuna
import torch
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset

# === GPU指定 ===
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# === 設定 ===
CORPUS_PATH = "../data/merged_blocks_50%.csv"
TOKENIZER_PATH = "tokenizer"
MODEL_OUTPUT_DIR = "output/bert_pretrain_optuna"
DB_PATH = "sqlite:///optuna.db"
STUDY_NAME = "bert_pretrain_gpu"

# === トークナイザーとデータ ===
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
dataset = load_dataset("text", data_files={"train": CORPUS_PATH})

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

tokenized = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# === Optuna評価関数 ===
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

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
        per_device_train_batch_size=16,
        num_train_epochs=1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        evaluation_strategy="no"
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

# === Studyロード & 実行 ===
study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
study.optimize(objective, n_trials=5)
