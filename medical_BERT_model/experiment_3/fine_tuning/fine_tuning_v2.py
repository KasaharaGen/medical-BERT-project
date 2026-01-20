#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fine_tuning.py
Teacher: microsoft/deberta-v2-xlarge (AutoModel, dtype="auto")
Student: BERT系 SequenceClassification
Tokenizer: 分離
KD: logits + representation (CLS)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import random
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

import matplotlib.pyplot as plt


# =========================
# 共通ユーティリティ
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# データ処理
# =========================
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {"text", "label"} <= set(df.columns)
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)
    return df


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    keep_text: bool,
) -> Dataset:
    ds = Dataset.from_pandas(df.reset_index(drop=True))

    def _map(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        if keep_text:
            out["text"] = batch["text"]
        return out

    remove_cols = [] if keep_text else ["text"]
    ds = ds.map(_map, batched=True, remove_columns=remove_cols)
    ds = ds.rename_column("label", "labels")
    return ds


class CollatorWithText:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        texts = None
        if "text" in features[0]:
            texts = [f["text"] for f in features]
            features = [{k: v for k, v in f.items() if k != "text"} for f in features]

        batch = self.tokenizer.pad(features, return_tensors="pt")
        if texts is not None:
            batch["text"] = texts
        return batch


# =========================
# Prior / Logit Adjustment
# =========================
def compute_prior_adjustment(labels: np.ndarray) -> float:
    p = labels.mean()
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log((1 - p) / p)


def apply_logit_adjustment(logits: torch.Tensor, adjust: float, tau: float):
    if tau == 0.0:
        return logits
    out = logits.clone()
    out[:, 1] -= tau * adjust
    return out


# =========================
# Teacher: DeBERTa v2 xlarge
# =========================
class DebertaTeacher(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype="auto",          # ← 指定どおり
        )
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits, out.hidden_states


# =========================
# Distillation Trainer
# =========================
class DistillTrainer(Trainer):
    def __init__(
        self,
        teacher_model: nn.Module,
        teacher_tokenizer,
        teacher_max_length: int,
        distill_alpha: float,
        temperature: float,
        rep_beta: float,
        prior_adjust: float,
        prior_tau: float,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_max_length = teacher_max_length

        self.alpha = distill_alpha
        self.T = temperature
        self.rep_beta = rep_beta
        self.prior_adjust = prior_adjust
        self.prior_tau = prior_tau

        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        texts = inputs["text"]

        student_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "text"]}
        out_s = model(**student_inputs, output_hidden_states=True)
        logits_s = out_s.logits

        device = logits_s.device
        self.teacher.to(device)

        teacher_batch = self.teacher_tokenizer(
            list(texts),
            truncation=True,
            max_length=self.teacher_max_length,
            padding=True,
            return_tensors="pt",
        )
        teacher_batch = {k: v.to(device) for k, v in teacher_batch.items()}

        with torch.no_grad():
            logits_t, hidden_t = self.teacher(
                teacher_batch["input_ids"],
                teacher_batch["attention_mask"],
            )

        logits_s = apply_logit_adjustment(logits_s, self.prior_adjust, self.prior_tau)
        logits_t = apply_logit_adjustment(logits_t, self.prior_adjust, self.prior_tau)

        ce = nn.CrossEntropyLoss()
        loss_ce = ce(logits_s, labels)

        log_p_s = torch.log_softmax(logits_s / self.T, dim=-1)
        p_t = torch.softmax(logits_t / self.T, dim=-1)
        loss_kd = self.kl(log_p_s, p_t) * (self.T ** 2)

        rep_s = out_s.hidden_states[-1][:, 0, :]
        rep_t = hidden_t[-1][:, 0, :]
        loss_rep = self.mse(rep_s, rep_t)

        loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd + self.rep_beta * loss_rep
        return (loss, out_s) if return_outputs else loss


# =========================
# 混同行列（％・Blues）
# =========================
def plot_confusion_matrix(cm, out_path):
    cm = cm.astype(float)
    cm = cm / cm.sum() * 100.0

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:.1f}%", ha="center", va="center", fontsize=16)

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--student_model_dir", required=True)
    parser.add_argument("--student_tokenizer_dir", required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--teacher_max_length", type=int, default=512)

    parser.add_argument("--distill_alpha", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--rep_beta", type=float, default=0.5)
    parser.add_argument("--prior_tau", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./result")

    args = parser.parse_args()
    set_all_seeds(args.seed)

    df = load_df(args.csv)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=args.seed
    )

    prior_adjust = compute_prior_adjustment(train_df["label"].values)

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer_dir)
    teacher_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")

    ds_train = tokenize_dataset(train_df, student_tokenizer, args.max_length, keep_text=True)
    ds_test = tokenize_dataset(test_df, student_tokenizer, args.max_length, keep_text=False)

    student_config = AutoConfig.from_pretrained(
        args.student_model_dir,
        num_labels=2,
        output_hidden_states=True,
    )
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model_dir,
        config=student_config,
    )

    teacher = DebertaTeacher("microsoft/deberta-v2-xlarge")

    collator = CollatorWithText(student_tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="no",
        save_strategy="no",
        fp16=True,
        report_to="none",
    )

    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=ds_train,
        tokenizer=student_tokenizer,
        data_collator=collator,

        teacher_model=teacher,
        teacher_tokenizer=teacher_tokenizer,
        teacher_max_length=args.teacher_max_length,

        distill_alpha=args.distill_alpha,
        temperature=args.temperature,
        rep_beta=args.rep_beta,
        prior_adjust=prior_adjust,
        prior_tau=args.prior_tau,
    )

    trainer.train()

    # Test
    device = get_device()
    student.to(device)
    student.eval()

    logits = []
    labels = []

    for batch in ds_test:
        inp = student_tokenizer.pad([batch], return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = student(**inp)
        logits.append(out.logits.cpu().numpy())
        labels.append(batch["labels"])

    logits = np.concatenate(logits)
    labels = np.array(labels)

    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(labels, preds)
    os.makedirs(args.output_dir, exist_ok=True)
    plot_confusion_matrix(cm, os.path.join(args.output_dir, "test_confusion_matrix_percent_blues.png"))

    print("Test MCC:", matthews_corrcoef(labels, preds))


if __name__ == "__main__":
    main()
