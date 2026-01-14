#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fine_tuning.py (tokenizer-separated teacher/student)

主目的:
- teacher と student のトークナイザーを分離して蒸留を成立させる
  (teacherにDeBERTa/RoBERTa/Longformer等の別アーキテクチャを入れても破綻しない)

継続仕様（これまでの要件）:
- 蒸留付きFT（logits蒸留 + 表現蒸留）
- 可能ならLoRA（studentのみ, peftがあれば）
- Validation(OOF)で閾値最適化 → Testは固定閾値
- Logit Adjustment / Prior補正
- クラス重み緩和（power/clip）
- Teacher強化（foldごと supervised FT）
- Stratified K-fold
- 混同行列: Blues統一 + %表記
- curve_loss.png, curve_eval_metrics.png 出力（final学習時）

重要な実装変更点:
- student学習データは "text" を保持し、DistillTrainer.compute_lossで teacher_tokenizer により
  その場で teacher 入力をtokenizeする（teacherとstudentでtokenizer不一致OK）
- Teacherの教師あり学習は teacher_tokenizer で通常通り tokenization（text保持不要）

注意:
- teacher側tokenizeを毎step行うため計算コストは増える（正しさ優先）
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import random
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import torch
from torch import nn

from datasets import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix

import optuna
from optuna.pruners import MedianPruner

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)

import matplotlib.pyplot as plt


# =========================
# ユーティリティ
# =========================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def must_dir(p: str, name: str) -> Path:
    path = Path(p)
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"{name} が存在しないかディレクトリではない: {path}")
    return path


def must_file(p: str, name: str) -> Path:
    path = Path(p)
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"{name} が存在しないかファイルではない: {path}")
    return path


def ensure_empty_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# データ
# =========================
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not ({"text", "label"} <= set(df.columns)):
        raise RuntimeError("CSVに text,label 列が必要である。")
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)
    return df


def stratified_test_split(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    keep_text: bool,
) -> Dataset:
    """
    keep_text=True : Datasetに text を残す（student蒸留で teacher_tokenizer を使うため）
    keep_text=False: textを削除して軽量化（teacher supervised 等）
    """
    ds = Dataset.from_pandas(df.reset_index(drop=True))

    def _tok_map(batch):
        tok = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        if keep_text:
            tok["text"] = batch["text"]
        return tok

    remove_cols = [] if keep_text else ["text"]
    ds = ds.map(_tok_map, batched=True, remove_columns=remove_cols)
    ds = ds.rename_column("label", "labels") if "label" in ds.column_names else ds
    return ds


# =========================
# Collator（text保持対応）
# =========================
class CollatorWithText:
    """
    Hugging Faceのpadを使いつつ、textがあれば保持して返す。
    """
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = None
        if len(features) > 0 and "text" in features[0]:
            texts = [f["text"] for f in features]
            features = [{k: v for k, v in f.items() if k != "text"} for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if texts is not None:
            batch["text"] = texts
        return batch


# =========================
# クラス重み（緩和/クリップ）
# =========================
def build_class_weight(
    train_labels: np.ndarray,
    power: float = 1.0,
    clip_max: float = 1e9,
) -> Optional[torch.Tensor]:
    pos = int(train_labels.sum())
    neg = int(len(train_labels) - pos)
    if pos == 0 or neg == 0:
        return None

    w0 = len(train_labels) / (2.0 * neg)
    w1 = len(train_labels) / (2.0 * pos)

    w0 = float(w0) ** float(power)
    w1 = float(w1) ** float(power)

    w0 = min(w0, clip_max)
    w1 = min(w1, clip_max)

    return torch.tensor([w0, w1], dtype=torch.float32)


# =========================
# Logit Adjustment / Prior補正
# =========================
def compute_prior_logit_adjustment(train_labels: np.ndarray) -> float:
    pi = float(np.mean(train_labels))
    pi = min(max(pi, 1e-6), 1.0 - 1e-6)
    return float(math.log((1.0 - pi) / pi))


def apply_logit_adjustment(logits: torch.Tensor, adjust: float, tau: float) -> torch.Tensor:
    if tau == 0.0:
        return logits
    out = logits.clone()
    out[:, 1] = out[:, 1] - float(tau) * float(adjust)
    return out


# =========================
# 閾値最適化（OOF/validation用）
# =========================
def sweep_threshold_best_mcc(
    prob1: np.ndarray,
    labels: np.ndarray,
    n_grid: int = 401,
) -> Dict[str, Any]:
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best = {"thr": 0.5, "mcc": -1.0, "f1": 0.0, "acc": 0.0, "cm": None}

    for thr in thresholds:
        preds = (prob1 >= thr).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best["mcc"]:
            best["mcc"] = float(mcc)
            best["thr"] = float(thr)
            best["f1"] = float(f1_score(labels, preds))
            best["acc"] = float(accuracy_score(labels, preds))
            best["cm"] = confusion_matrix(labels, preds, labels=[0, 1])

    if best["cm"] is None:
        preds = (prob1 >= 0.5).astype(int)
        best["thr"] = 0.5
        best["acc"] = float(accuracy_score(labels, preds))
        if len(np.unique(preds)) > 1:
            best["mcc"] = float(matthews_corrcoef(labels, preds))
            best["f1"] = float(f1_score(labels, preds))
        else:
            best["mcc"] = 0.0
            best["f1"] = 0.0
        best["cm"] = confusion_matrix(labels, preds, labels=[0, 1])

    return best


def logits_to_prob1(logits_np: np.ndarray) -> np.ndarray:
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    prob1 = torch.softmax(logits_t, dim=-1)[:, 1].cpu().numpy()
    return prob1


# =========================
# 推論（ragged回避）
# =========================
@torch.no_grad()
def predict_logits_with_dataloader(
    model: nn.Module,
    dataset: Dataset,
    collator,
    batch_size: int,
    amp: str = "fp16",
) -> Tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    device = get_device()
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    use_amp = torch.cuda.is_available() and (amp in ["fp16", "bf16"])
    autocast_dtype = torch.float16 if amp == "fp16" else (torch.bfloat16 if amp == "bf16" else None)

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
        for batch in loader:
            # textが混じっていても無視する
            if "text" in batch:
                batch = {k: v for k, v in batch.items() if k != "text"}

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.detach().cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels)

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels


# =========================
# 曲線プロット（loss/metrics）
# =========================
def plot_curve_loss(history: List[Dict[str, Any]], out_path: str) -> None:
    steps = [h["step"] for h in history]
    train_loss = [h.get("train_loss", None) for h in history]
    eval_loss = [h.get("eval_loss", None) for h in history]

    plt.figure(figsize=(9, 5))
    if any(v is not None for v in train_loss):
        xs = [s for s, v in zip(steps, train_loss) if v is not None]
        ys = [v for v in train_loss if v is not None]
        plt.plot(xs, ys, label="train_loss")
    if any(v is not None for v in eval_loss):
        xs = [s for s, v in zip(steps, eval_loss) if v is not None]
        ys = [v for v in eval_loss if v is not None]
        plt.plot(xs, ys, label="eval_loss")

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_curve_eval_metrics(history: List[Dict[str, Any]], out_path: str) -> None:
    steps = [h["step"] for h in history]

    def _series(key):
        xs = [s for s, h in zip(steps, history) if h.get(key, None) is not None]
        ys = [h[key] for h in history if h.get(key, None) is not None]
        return xs, ys

    plt.figure(figsize=(9, 5))
    for key in ["eval_mcc", "eval_f1", "eval_acc"]:
        xs, ys = _series(key)
        if len(xs) > 0:
            plt.plot(xs, ys, label=key)

    plt.xlabel("step")
    plt.ylabel("score")
    plt.title("Eval Metrics Curve (threshold-optimized on eval)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# LoRA（任意、studentのみ）
# =========================
def maybe_apply_lora_to_student(
    model: nn.Module,
    enabled: bool,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Tuple[str, ...],
) -> Tuple[nn.Module, bool]:
    if not enabled:
        return model, False
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        return model, False

    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, cfg)
    return model, True


def force_classifier_trainable(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True


# =========================
# 蒸留Trainer（tokenizer分離版）
# =========================
class DistillTrainer(Trainer):
    """
    loss = (1-alpha)*CE + alpha*T^2*KL(teacher||student) + beta*MSE(CLS_teacher, CLS_student)

    重要:
    - inputs に "text"（list[str]）を含める
    - teacherは teacher_tokenizer でこのtextをtokenizeしてforwardする
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        teacher_tokenizer,
        teacher_max_length: int,
        distill_alpha: float,
        temperature: float,
        rep_beta: float,
        class_weight: Optional[torch.Tensor],
        label_smoothing: float,
        prior_adjust: float,
        prior_tau: float,
        amp: str,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_max_length = int(teacher_max_length)

        self.distill_alpha = float(distill_alpha)
        self.temperature = float(temperature)
        self.rep_beta = float(rep_beta)
        self.class_weight = class_weight
        self.label_smoothing = float(label_smoothing)
        self.prior_adjust = float(prior_adjust)
        self.prior_tau = float(prior_tau)
        self.amp = amp

        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss(reduction="mean")

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # student inputs（textは除く）
        texts = inputs.get("text", None)
        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "text"]}

        out_s = model(**model_inputs, output_hidden_states=True)
        logits_s = out_s.logits
        device = logits_s.device

        # teacherを同deviceへ
        t_dev = next(self.teacher_model.parameters()).device
        if t_dev != device:
            self.teacher_model.to(device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # teacher inputs（teacher_tokenizerでtextをtokenize）
        if texts is None:
            raise RuntimeError("tokenizer分離蒸留には inputs['text'] が必要である（Dataset/collatorを確認）。")

        teacher_batch = self.teacher_tokenizer(
            list(texts),
            truncation=True,
            max_length=self.teacher_max_length,
            padding=True,
            return_tensors="pt",
        )
        teacher_batch = {k: v.to(device) for k, v in teacher_batch.items()}

        with torch.no_grad():
            out_t = self.teacher_model(**teacher_batch, output_hidden_states=True)
            logits_t = out_t.logits

        # prior補正
        logits_s_adj = apply_logit_adjustment(logits_s, self.prior_adjust, self.prior_tau)
        logits_t_adj = apply_logit_adjustment(logits_t, self.prior_adjust, self.prior_tau)

        # CE
        weight = self.class_weight.to(device) if self.class_weight is not None else None
        ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss_ce = ce(logits_s_adj.view(-1, 2), labels.view(-1))

        # KD
        T = self.temperature
        log_p_s = torch.log_softmax(logits_s_adj / T, dim=-1)
        p_t = torch.softmax(logits_t_adj / T, dim=-1)
        loss_kd = self.kl(log_p_s, p_t) * (T * T)

        # 表現蒸留（最後層CLS）
        hs_s = out_s.hidden_states[-1][:, 0, :]
        hs_t = out_t.hidden_states[-1][:, 0, :]
        loss_rep = self.mse(hs_s, hs_t)

        a = self.distill_alpha
        b = self.rep_beta
        loss = (1.0 - a) * loss_ce + a * loss_kd + b * loss_rep

        return (loss, out_s) if return_outputs else loss


# =========================
# 曲線出力Callback（final学習で使用）
# =========================
class CurveLoggerCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dataset,
        out_dir: str,
        amp: str,
        batch_size: int,
        threshold_grid: int,
        prior_adjust: float,
        prior_tau: float,
        collator,
        prefix: str = "",
    ):
        self.eval_dataset = eval_dataset
        self.out_dir = out_dir
        self.amp = amp
        self.batch_size = batch_size
        self.threshold_grid = threshold_grid
        self.prior_adjust = float(prior_adjust)
        self.prior_tau = float(prior_tau)
        self.collator = collator
        self.prefix = prefix

        self.history: List[Dict[str, Any]] = []
        os.makedirs(self.out_dir, exist_ok=True)

    def _ensure_row(self, step: int) -> Dict[str, Any]:
        for h in reversed(self.history):
            if h.get("step") == step:
                return h
        row = {"step": int(step)}
        self.history.append(row)
        return row

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        row = self._ensure_row(step)
        if "loss" in logs and logs["loss"] is not None:
            row["train_loss"] = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        step = int(state.global_step)
        row = self._ensure_row(step)

        if metrics and "eval_loss" in metrics and metrics["eval_loss"] is not None:
            row["eval_loss"] = float(metrics["eval_loss"])

        if model is None:
            return

        logits, labels = predict_logits_with_dataloader(
            model=model,
            dataset=self.eval_dataset,
            collator=self.collator,
            batch_size=max(8, int(self.batch_size)),
            amp=self.amp,
        )
        logits_t = torch.tensor(logits, dtype=torch.float32)
        logits_adj = apply_logit_adjustment(logits_t, self.prior_adjust, self.prior_tau).numpy()
        prob1 = logits_to_prob1(logits_adj)

        best = sweep_threshold_best_mcc(prob1, labels, n_grid=int(self.threshold_grid))
        row["eval_mcc"] = float(best["mcc"])
        row["eval_f1"] = float(best["f1"])
        row["eval_acc"] = float(best["acc"])
        row["eval_thr"] = float(best["thr"])

    def on_train_end(self, args, state, control, **kwargs):
        hist_path = os.path.join(self.out_dir, f"{self.prefix}curve_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        loss_path = os.path.join(self.out_dir, f"{self.prefix}curve_loss.png")
        eval_path = os.path.join(self.out_dir, f"{self.prefix}curve_eval_metrics.png")
        plot_curve_loss(self.history, loss_path)
        plot_curve_eval_metrics(self.history, eval_path)


# =========================
# 混同行列プロット（Blues + %表記）
# =========================
def plot_confusion_matrix_percent_blues(
    cm: np.ndarray,
    out_path: str,
    title: str,
    percent_mode: str = "all",
    fontsize: int = 18,
) -> None:
    cm = np.array(cm, dtype=float)
    if percent_mode == "true":
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_pct = (cm / row_sum) * 100.0
    else:
        denom = cm.sum()
        denom = denom if denom > 0 else 1.0
        cm_pct = (cm / denom) * 100.0

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=max(1.0, cm_pct.max()))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred_0", "Pred_1"])
    plt.yticks(tick_marks, ["True_0", "True_1"])

    thresh = cm_pct.max() / 2.0 if cm_pct.size > 0 else 0.0
    for i in range(2):
        for j in range(2):
            val = cm_pct[i, j]
            plt.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=fontsize, fontweight="bold",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# Teacher学習（強化）
# =========================
def train_teacher_supervised(
    args,
    teacher_tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    trial_params: Dict[str, Any],
    save_model: bool,
) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
    max_length = int(trial_params.get("teacher_max_length", args.teacher_max_length))
    ds_tr = tokenize_dataset(train_df, teacher_tokenizer, max_length=max_length, keep_text=False)
    ds_va = tokenize_dataset(val_df, teacher_tokenizer, max_length=max_length, keep_text=False)

    train_labels = train_df["label"].values
    cw = None
    if args.use_class_weight:
        cw = build_class_weight(
            train_labels,
            power=float(trial_params.get("class_weight_power", args.class_weight_power)),
            clip_max=float(trial_params.get("class_weight_clip", args.class_weight_clip)),
        )

    teacher_config = AutoConfig.from_pretrained(
        args.teacher_base_dir,
        num_labels=2,
        hidden_dropout_prob=float(trial_params.get("teacher_hidden_dropout_prob", args.teacher_hidden_dropout_prob)),
        attention_probs_dropout_prob=float(trial_params.get("teacher_attention_dropout_prob", args.teacher_attention_dropout_prob)),
        classifier_dropout=float(trial_params.get("teacher_classifier_dropout", args.teacher_classifier_dropout)),
        output_hidden_states=True,
    )

    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_base_dir, config=teacher_config)

    collator = CollatorWithText(teacher_tokenizer, pad_to_multiple_of=8)

    amp = args.amp
    fp16 = (amp == "fp16")
    bf16 = (amp == "bf16")

    if save_model:
        save_strategy = "steps"
        load_best = True
        save_steps = int(args.eval_steps)
        save_total_limit = 1
    else:
        save_strategy = "no"
        load_best = False
        save_steps = None
        save_total_limit = 0

    tr_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=float(trial_params.get("teacher_epochs", args.teacher_epochs)),
        per_device_train_batch_size=int(trial_params.get("batch_size", args.batch_size)),
        per_device_eval_batch_size=max(8, int(trial_params.get("batch_size", args.batch_size))),
        gradient_accumulation_steps=int(trial_params.get("grad_accum", args.grad_accum)),
        learning_rate=float(trial_params.get("teacher_lr", args.teacher_lr)),
        weight_decay=float(trial_params.get("teacher_weight_decay", args.teacher_weight_decay)),
        warmup_ratio=float(trial_params.get("warmup_ratio", args.warmup_ratio)),
        lr_scheduler_type=str(trial_params.get("lr_scheduler_type", args.lr_scheduler_type)),
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=int(args.seed),
        report_to=["none"],
        dataloader_num_workers=4,
        fp16=fp16,
        bf16=bf16,
        disable_tqdm=True,
    )

    class TeacherTrainer(Trainer):
        def __init__(self, class_weight=None, label_smoothing=0.0, *a, **kw):
            super().__init__(*a, **kw)
            self.class_weight = class_weight
            self.label_smoothing = float(label_smoothing)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "text"]}
            out = model(**model_inputs)
            logits = out.logits
            device = logits.device
            w = self.class_weight.to(device) if self.class_weight is not None else None
            ce = nn.CrossEntropyLoss(weight=w, label_smoothing=self.label_smoothing)
            loss = ce(logits.view(-1, 2), labels.view(-1))
            return (loss, out) if return_outputs else loss

    teacher_trainer = TeacherTrainer(
        model=teacher,
        args=tr_args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=teacher_tokenizer,
        data_collator=collator,
        class_weight=cw,
        label_smoothing=float(trial_params.get("teacher_label_smoothing", args.teacher_label_smoothing)),
    )

    teacher_trainer.train()
    eval_metrics = teacher_trainer.evaluate()

    if save_model:
        os.makedirs(output_dir, exist_ok=True)
        teacher_trainer.save_model(os.path.join(output_dir, "teacher_model"))

    return teacher, {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float, np.number))}


# =========================
# Student蒸留学習（fold単位）
# =========================
def train_student_distill(
    args,
    student_tokenizer,
    teacher_tokenizer,
    teacher_model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    trial_params: Dict[str, Any],
    prior_adjust: float,
    save_model: bool,
    curve_out_dir: Optional[str] = None,
    curve_prefix: str = "",
) -> Tuple[AutoModelForSequenceClassification, Dict[str, float], np.ndarray, np.ndarray]:

    student_max_length = int(trial_params.get("max_length", args.max_length))
    teacher_max_length = int(trial_params.get("teacher_max_length", args.teacher_max_length))

    # student側は text を保持（teacher tokenize用）
    ds_tr = tokenize_dataset(train_df, student_tokenizer, max_length=student_max_length, keep_text=True)
    ds_va = tokenize_dataset(val_df, student_tokenizer, max_length=student_max_length, keep_text=True)

    train_labels = train_df["label"].values
    cw = None
    if args.use_class_weight:
        cw = build_class_weight(
            train_labels,
            power=float(trial_params.get("class_weight_power", args.class_weight_power)),
            clip_max=float(trial_params.get("class_weight_clip", args.class_weight_clip)),
        )

    student_config = AutoConfig.from_pretrained(
        args.student_model_dir,
        num_labels=2,
        hidden_dropout_prob=float(trial_params.get("hidden_dropout_prob", args.hidden_dropout_prob)),
        attention_probs_dropout_prob=float(trial_params.get("attention_probs_dropout_prob", args.attention_probs_dropout_prob)),
        classifier_dropout=float(trial_params.get("classifier_dropout", args.classifier_dropout)),
        output_hidden_states=True,
    )

    student = AutoModelForSequenceClassification.from_pretrained(args.student_model_dir, config=student_config)

    use_lora = bool(trial_params.get("use_lora", args.use_lora))
    student, lora_on = maybe_apply_lora_to_student(
        student,
        enabled=use_lora,
        r=int(trial_params.get("lora_r", args.lora_r)),
        alpha=int(trial_params.get("lora_alpha", args.lora_alpha)),
        dropout=float(trial_params.get("lora_dropout", args.lora_dropout)),
        target_modules=tuple(str(trial_params.get("lora_targets", args.lora_targets)).split(",")),
    )
    if use_lora and (not lora_on):
        print("[WARN] peft未導入のため LoRA をスキップした（full FTになる）")

    force_classifier_trainable(student)

    collator_student = CollatorWithText(student_tokenizer, pad_to_multiple_of=8)

    amp = args.amp
    fp16 = (amp == "fp16")
    bf16 = (amp == "bf16")

    if save_model:
        save_strategy = "steps"
        load_best = True
        save_steps = int(args.eval_steps)
        save_total_limit = 1
    else:
        save_strategy = "no"
        load_best = False
        save_steps = None
        save_total_limit = 0

    tr_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=float(trial_params.get("epochs", args.epochs)),
        per_device_train_batch_size=int(trial_params.get("batch_size", args.batch_size)),
        per_device_eval_batch_size=max(8, int(trial_params.get("batch_size", args.batch_size))),
        gradient_accumulation_steps=int(trial_params.get("grad_accum", args.grad_accum)),
        learning_rate=float(trial_params.get("lr", args.lr)),
        weight_decay=float(trial_params.get("weight_decay", args.weight_decay)),
        warmup_ratio=float(trial_params.get("warmup_ratio", args.warmup_ratio)),
        lr_scheduler_type=str(trial_params.get("lr_scheduler_type", args.lr_scheduler_type)),
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=int(args.seed),
        report_to=["none"],
        dataloader_num_workers=4,
        fp16=fp16,
        bf16=bf16,
        disable_tqdm=True,
    )

    trainer = DistillTrainer(
        model=student,
        args=tr_args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=student_tokenizer,
        data_collator=collator_student,

        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        teacher_max_length=teacher_max_length,

        distill_alpha=float(trial_params.get("distill_alpha", args.distill_alpha)),
        temperature=float(trial_params.get("temperature", args.temperature)),
        rep_beta=float(trial_params.get("rep_beta", args.rep_beta)),
        class_weight=cw,
        label_smoothing=float(trial_params.get("label_smoothing", args.label_smoothing)),
        prior_adjust=prior_adjust,
        prior_tau=float(trial_params.get("prior_tau", args.prior_tau)),
        amp=args.amp,
    )

    if curve_out_dir is not None:
        trainer.add_callback(
            CurveLoggerCallback(
                eval_dataset=ds_va,
                out_dir=curve_out_dir,
                amp=args.amp,
                batch_size=int(trial_params.get("batch_size", args.batch_size)),
                threshold_grid=int(args.threshold_grid),
                prior_adjust=prior_adjust,
                prior_tau=float(trial_params.get("prior_tau", args.prior_tau)),
                collator=collator_student,
                prefix=curve_prefix,
            )
        )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # validation logits（studentのみ。teacher tokenizer不要）
    val_logits, val_labels = predict_logits_with_dataloader(
        model=trainer.model,
        dataset=ds_va.remove_columns(["text"]),
        collator=CollatorWithText(student_tokenizer, pad_to_multiple_of=8),
        batch_size=max(8, int(trial_params.get("batch_size", args.batch_size))),
        amp=args.amp,
    )

    val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
    val_logits_adj = apply_logit_adjustment(val_logits_t, prior_adjust, float(trial_params.get("prior_tau", args.prior_tau))).numpy()
    val_prob1 = logits_to_prob1(val_logits_adj)

    if save_model:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(os.path.join(output_dir, "student_model"))

    eval_out = {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float, np.number))}
    return student, eval_out, val_prob1, val_labels


# =========================
# K-fold（OOFで閾値最適化）
# =========================
def kfold_oof_train_eval(
    args,
    df_trainval: pd.DataFrame,
    student_tokenizer,
    teacher_tokenizer,
    trial_params: Dict[str, Any],
    save_models: bool,
    work_dir: str,
) -> Dict[str, Any]:
    labels = df_trainval["label"].values
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    prior_adjust = compute_prior_logit_adjustment(labels)

    oof_prob1 = np.zeros(len(df_trainval), dtype=float)
    oof_labels = labels.copy()

    fold_metrics: List[Dict[str, Any]] = []
    fold_best_mccs: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df_trainval)), labels), start=1):
        train_df = df_trainval.iloc[tr_idx].reset_index(drop=True)
        val_df = df_trainval.iloc[va_idx].reset_index(drop=True)

        fold_dir = os.path.join(work_dir, f"fold_{fold:02d}")
        ensure_empty_dir(fold_dir)

        teacher_dir = os.path.join(fold_dir, "teacher")
        student_dir = os.path.join(fold_dir, "student")
        os.makedirs(teacher_dir, exist_ok=True)
        os.makedirs(student_dir, exist_ok=True)

        teacher_model, teacher_eval = train_teacher_supervised(
            args=args,
            teacher_tokenizer=teacher_tokenizer,
            train_df=train_df,
            val_df=val_df,
            output_dir=teacher_dir,
            trial_params=trial_params,
            save_model=save_models,
        )

        _, student_eval, val_prob1, val_labels = train_student_distill(
            args=args,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            teacher_model=teacher_model,
            train_df=train_df,
            val_df=val_df,
            output_dir=student_dir,
            trial_params=trial_params,
            prior_adjust=prior_adjust,
            save_model=save_models,
            curve_out_dir=None,
        )

        best_fold = sweep_threshold_best_mcc(val_prob1, val_labels, n_grid=args.threshold_grid)
        fold_best_mccs.append(float(best_fold["mcc"]))

        oof_prob1[va_idx] = val_prob1

        fold_metrics.append({
            "fold": fold,
            "teacher_eval": teacher_eval,
            "student_eval": student_eval,
            "fold_best": {
                "thr": float(best_fold["thr"]),
                "mcc": float(best_fold["mcc"]),
                "f1": float(best_fold["f1"]),
                "acc": float(best_fold["acc"]),
                "cm": best_fold["cm"].tolist() if best_fold["cm"] is not None else None,
            }
        })

        del teacher_model
        torch.cuda.empty_cache()

    oof_best = sweep_threshold_best_mcc(oof_prob1, oof_labels, n_grid=args.threshold_grid)

    return {
        "oof_best": {
            "thr": float(oof_best["thr"]),
            "mcc": float(oof_best["mcc"]),
            "f1": float(oof_best["f1"]),
            "acc": float(oof_best["acc"]),
            "cm": oof_best["cm"].tolist() if oof_best["cm"] is not None else None,
        },
        "fold_mean_mcc": float(np.mean(fold_best_mccs)) if len(fold_best_mccs) else 0.0,
        "fold_metrics": fold_metrics,
        "best_threshold": float(oof_best["thr"]),
        "prior_adjust": float(prior_adjust),
    }


# =========================
# Final: test評価（固定thr）
# =========================
@torch.no_grad()
def evaluate_on_test_fixed_threshold(
    args,
    student_tokenizer,
    student_model_dir: str,
    test_df: pd.DataFrame,
    max_length: int,
    prior_adjust: float,
    prior_tau: float,
    fixed_thr: float,
    out_dir: str,
) -> Dict[str, Any]:
    ds_te = tokenize_dataset(test_df, student_tokenizer, max_length=max_length, keep_text=False)
    collator = CollatorWithText(student_tokenizer, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(student_model_dir)
    model.eval()

    logits, labels = predict_logits_with_dataloader(
        model=model,
        dataset=ds_te,
        collator=collator,
        batch_size=args.batch_size,
        amp=args.amp,
    )

    logits_t = torch.tensor(logits, dtype=torch.float32)
    logits_adj = apply_logit_adjustment(logits_t, prior_adjust, prior_tau).numpy()

    prob1 = logits_to_prob1(logits_adj)
    preds = (prob1 >= fixed_thr).astype(int)

    mcc = float(matthews_corrcoef(labels, preds)) if len(np.unique(preds)) > 1 else 0.0
    f1 = float(f1_score(labels, preds)) if len(np.unique(preds)) > 1 else 0.0
    acc = float(accuracy_score(labels, preds))
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics_fixed_threshold.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "fixed_threshold": float(fixed_thr),
                "mcc": mcc,
                "f1": f1,
                "accuracy": acc,
                "cm": cm.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_confusion_matrix_percent_blues(
        cm=cm,
        out_path=os.path.join(out_dir, "test_confusion_matrix_percent_blues.png"),
        title=f"Test Confusion Matrix (fixed thr={fixed_thr:.2f})",
        percent_mode=args.cm_percent_mode,
        fontsize=20,
    )

    pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"]).to_csv(
        os.path.join(out_dir, "test_confusion_matrix_counts.csv")
    )

    cm_pct = cm.astype(float)
    if args.cm_percent_mode == "true":
        rs = cm_pct.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        cm_pct = (cm_pct / rs) * 100.0
    else:
        denom = cm_pct.sum()
        denom = denom if denom > 0 else 1.0
        cm_pct = (cm_pct / denom) * 100.0

    pd.DataFrame(cm_pct, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"]).to_csv(
        os.path.join(out_dir, "test_confusion_matrix_percent.csv")
    )

    return {"fixed_threshold": float(fixed_thr), "mcc": mcc, "f1": f1, "accuracy": acc, "cm": cm.tolist()}


def final_train_and_test(
    args,
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    student_tokenizer,
    teacher_tokenizer,
    trial_params: Dict[str, Any],
    best_threshold: float,
    prior_adjust: float,
    out_dir: str,
) -> Dict[str, Any]:
    student_max_length = int(trial_params.get("max_length", args.max_length))

    ensure_empty_dir(out_dir)
    teacher_dir = os.path.join(out_dir, "teacher_final")
    student_dir = os.path.join(out_dir, "best_model")
    tok_dir = os.path.join(out_dir, "tokenizer")
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    tv_train, tv_val = train_test_split(
        df_trainval,
        test_size=min(0.1, args.val_ratio_for_final),
        random_state=args.seed,
        stratify=df_trainval["label"],
        shuffle=True,
    )
    tv_train = tv_train.reset_index(drop=True)
    tv_val = tv_val.reset_index(drop=True)

    teacher_model, teacher_eval = train_teacher_supervised(
        args=args,
        teacher_tokenizer=teacher_tokenizer,
        train_df=tv_train,
        val_df=tv_val,
        output_dir=teacher_dir,
        trial_params=trial_params,
        save_model=True,
    )

    student_work_dir = os.path.join(out_dir, "student_final_work")
    _, student_eval, _, _ = train_student_distill(
        args=args,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        teacher_model=teacher_model,
        train_df=tv_train,
        val_df=tv_val,
        output_dir=student_work_dir,
        trial_params=trial_params,
        prior_adjust=prior_adjust,
        save_model=True,
        curve_out_dir=out_dir,   # curve_loss.png / curve_eval_metrics.png
        curve_prefix="",
    )

    src_student = os.path.join(student_work_dir, "student_model")
    if not os.path.isdir(src_student):
        raise RuntimeError(f"studentモデルが見つからない: {src_student}")
    shutil.rmtree(student_dir, ignore_errors=True)
    shutil.copytree(src_student, student_dir)

    student_tokenizer.save_pretrained(tok_dir)

    with open(os.path.join(out_dir, "final_train_params.json"), "w", encoding="utf-8") as f:
        json.dump(trial_params, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "final_teacher_eval.json"), "w", encoding="utf-8") as f:
        json.dump(teacher_eval, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "final_student_eval.json"), "w", encoding="utf-8") as f:
        json.dump(student_eval, f, indent=2, ensure_ascii=False)

    test_metrics = evaluate_on_test_fixed_threshold(
        args=args,
        student_tokenizer=student_tokenizer,
        student_model_dir=student_dir,
        test_df=df_test,
        max_length=student_max_length,
        prior_adjust=prior_adjust,
        prior_tau=float(trial_params.get("prior_tau", args.prior_tau)),
        fixed_thr=best_threshold,
        out_dir=out_dir,
    )

    return test_metrics


# =========================
# Optuna（K-fold objective）
# =========================
def run_optuna_kfold(args, df_trainval: pd.DataFrame, student_tokenizer, teacher_tokenizer) -> Dict[str, Any]:
    os.makedirs(args.study_dir, exist_ok=True)

    pruner = MedianPruner(n_startup_trials=max(5, args.n_startup_trials)) if args.use_pruner else optuna.pruners.NopPruner()
    storage = args.optuna_storage if args.optuna_storage else None

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )

    trials_log_path = os.path.join(args.study_dir, "trials_log.csv")
    if not os.path.exists(trials_log_path):
        pd.DataFrame(columns=["trial", "oof_mcc", "fold_mean_mcc", "best_thr", "params_json"]).to_csv(trials_log_path, index=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 5e-6, 3e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 5e-2),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
            "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.2),
            "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.4),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "grad_accum": trial.suggest_categorical("grad_accum", [1, 2, 4, 8]),
            "epochs": args.epochs,
            "max_length": args.max_length,
            "lr_scheduler_type": args.lr_scheduler_type,

            "distill_alpha": trial.suggest_float("distill_alpha", 0.15, 0.55),
            "temperature": trial.suggest_float("temperature", 1.5, 4.0),
            "rep_beta": trial.suggest_float("rep_beta", 0.0, 2.0),

            "prior_tau": trial.suggest_float("prior_tau", 0.0, 2.0),

            "class_weight_power": trial.suggest_float("class_weight_power", 0.3, 1.0),
            "class_weight_clip": trial.suggest_float("class_weight_clip", 1.5, 5.0),

            "teacher_lr": trial.suggest_float("teacher_lr", 5e-6, 5e-5, log=True),
            "teacher_weight_decay": trial.suggest_float("teacher_weight_decay", 0.0, 5e-2),
            "teacher_epochs": args.teacher_epochs,
            "teacher_label_smoothing": trial.suggest_float("teacher_label_smoothing", 0.0, 0.1),

            "teacher_hidden_dropout_prob": trial.suggest_float("teacher_hidden_dropout_prob", 0.0, 0.2),
            "teacher_classifier_dropout": trial.suggest_float("teacher_classifier_dropout", 0.0, 0.3),
            "teacher_attention_dropout_prob": args.teacher_attention_dropout_prob,
            "attention_probs_dropout_prob": args.attention_probs_dropout_prob,

            "teacher_max_length": args.teacher_max_length,
        }

        params["use_lora"] = bool(args.use_lora)
        params["lora_targets"] = args.lora_targets
        if params["use_lora"]:
            params["lora_r"] = trial.suggest_categorical("lora_r", [4, 8, 16])
            if params["lora_r"] == 4:
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [8, 16])
            elif params["lora_r"] == 8:
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [16, 32])
            else:
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [32, 64])
            params["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.1)

        tmp_dir = os.path.join(args.study_dir, "_tmp_kfold")
        ensure_empty_dir(tmp_dir)

        res = kfold_oof_train_eval(
            args=args,
            df_trainval=df_trainval,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            trial_params=params,
            save_models=False,
            work_dir=tmp_dir,
        )

        oof_mcc = float(res["oof_best"]["mcc"])

        trial.report(oof_mcc, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        df_log = pd.read_csv(trials_log_path)
        df_log = pd.concat([df_log, pd.DataFrame([{
            "trial": int(trial.number),
            "oof_mcc": float(res["oof_best"]["mcc"]),
            "fold_mean_mcc": float(res["fold_mean_mcc"]),
            "best_thr": float(res["best_threshold"]),
            "params_json": json.dumps(params, ensure_ascii=False),
        }])], ignore_index=True)
        df_log.to_csv(trials_log_path, index=False)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return oof_mcc

    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": dict(study.best_trial.params),
    }

    merged = {
        "epochs": args.epochs,
        "max_length": args.max_length,
        "teacher_max_length": args.teacher_max_length,
        "lr_scheduler_type": args.lr_scheduler_type,
        "teacher_epochs": args.teacher_epochs,
        "teacher_attention_dropout_prob": args.teacher_attention_dropout_prob,
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "use_lora": bool(args.use_lora),
        "lora_targets": args.lora_targets,
    }
    merged.update(best["best_params"])

    out = {"best": best, "merged_best_params": merged}
    with open(os.path.join(args.study_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--student_model_dir", type=str, required=True)
    p.add_argument("--teacher_base_dir", type=str, required=True)

    # student tokenizer（従来のtokenizer_dir）
    p.add_argument("--student_tokenizer_dir", type=str, required=True)

    # teacher tokenizer（指定しなければteacher_base_dirを使う）
    p.add_argument("--teacher_tokenizer_dir", type=str, default="")

    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--test_ratio", type=float, default=0.2)

    p.add_argument("--output_dir", type=str, default="./result_distill_kfold")
    p.add_argument("--study_dir", type=str, default="./optuna_study")

    p.add_argument("--use_kfold", action="store_true", default=True)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--threshold_grid", type=int, default=401)
    p.add_argument("--val_ratio_for_final", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--teacher_max_length", type=int, default=512)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=3e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine"])

    p.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    p.add_argument("--attention_probs_dropout_prob", type=float, default=0.0)
    p.add_argument("--classifier_dropout", type=float, default=0.1)

    p.add_argument("--distill_alpha", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--rep_beta", type=float, default=0.5)

    p.add_argument("--use_class_weight", action="store_true", default=True)
    p.add_argument("--class_weight_power", type=float, default=0.5)
    p.add_argument("--class_weight_clip", type=float, default=3.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)

    p.add_argument("--prior_tau", type=float, default=1.0)

    p.add_argument("--teacher_epochs", type=int, default=3)
    p.add_argument("--teacher_lr", type=float, default=2e-5)
    p.add_argument("--teacher_weight_decay", type=float, default=3e-3)
    p.add_argument("--teacher_label_smoothing", type=float, default=0.0)
    p.add_argument("--teacher_hidden_dropout_prob", type=float, default=0.1)
    p.add_argument("--teacher_classifier_dropout", type=float, default=0.1)
    p.add_argument("--teacher_attention_dropout_prob", type=float, default=0.0)

    p.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=50)

    p.add_argument("--cm_percent_mode", type=str, default="all", choices=["all", "true"])

    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="query,key,value,dense")

    p.add_argument("--use_optuna", action="store_true", default=False)
    p.add_argument("--study_name", type=str, default="kd_seqcls_kfold")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_startup_trials", type=int, default=5)
    p.add_argument("--use_pruner", action="store_true", default=True)
    p.add_argument("--optuna_storage", type=str, default="")

    p.add_argument("--final_train_best", action="store_true", default=True)

    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    must_dir(args.student_model_dir, "STUDENT_MODEL_DIR")
    must_dir(args.teacher_base_dir, "TEACHER_BASE_DIR")
    must_dir(args.student_tokenizer_dir, "STUDENT_TOKENIZER_DIR")
    must_file(args.csv, "CSV")

    if args.teacher_tokenizer_dir.strip() != "":
        must_dir(args.teacher_tokenizer_dir, "TEACHER_TOKENIZER_DIR")

    if args.optuna_storage == "":
        args.optuna_storage = None

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.study_dir, exist_ok=True)

    # tokenizer分離
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer_dir, use_fast=True)
    teacher_tok_src = args.teacher_tokenizer_dir if args.teacher_tokenizer_dir.strip() != "" else args.teacher_base_dir
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tok_src, use_fast=True)

    df = load_df(args.csv)
    df_trainval, df_test = stratified_test_split(df, test_ratio=args.test_ratio, seed=args.seed)

    if args.use_optuna:
        best_info = run_optuna_kfold(args, df_trainval, student_tokenizer, teacher_tokenizer)
        trial_params = best_info["merged_best_params"]
    else:
        trial_params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "label_smoothing": args.label_smoothing,
            "hidden_dropout_prob": args.hidden_dropout_prob,
            "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
            "classifier_dropout": args.classifier_dropout,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "lr_scheduler_type": args.lr_scheduler_type,
            "distill_alpha": args.distill_alpha,
            "temperature": args.temperature,
            "rep_beta": args.rep_beta,
            "prior_tau": args.prior_tau,
            "class_weight_power": args.class_weight_power,
            "class_weight_clip": args.class_weight_clip,
            "teacher_lr": args.teacher_lr,
            "teacher_weight_decay": args.teacher_weight_decay,
            "teacher_epochs": args.teacher_epochs,
            "teacher_label_smoothing": args.teacher_label_smoothing,
            "teacher_hidden_dropout_prob": args.teacher_hidden_dropout_prob,
            "teacher_classifier_dropout": args.teacher_classifier_dropout,
            "teacher_attention_dropout_prob": args.teacher_attention_dropout_prob,
            "teacher_max_length": args.teacher_max_length,
            "use_lora": bool(args.use_lora),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
        }

    # k-foldでOOF閾値決定
    kfold_dir = os.path.join(args.output_dir, "kfold_oof")
    ensure_empty_dir(kfold_dir)

    res = kfold_oof_train_eval(
        args=args,
        df_trainval=df_trainval,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        trial_params=trial_params,
        save_models=False,
        work_dir=kfold_dir,
    )

    best_thr = float(res["best_threshold"])
    prior_adjust = float(res["prior_adjust"])

    with open(os.path.join(args.output_dir, "oof_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "chosen_threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"fixed_threshold": best_thr, "prior_adjust": prior_adjust}, f, indent=2, ensure_ascii=False)

    if args.final_train_best:
        final_dir = os.path.join(args.output_dir, "final")
        test_metrics = final_train_and_test(
            args=args,
            df_trainval=df_trainval,
            df_test=df_test,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            trial_params=trial_params,
            best_threshold=best_thr,
            prior_adjust=prior_adjust,
            out_dir=final_dir,
        )
        with open(os.path.join(final_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"fixed_threshold": best_thr, "prior_adjust": prior_adjust, "test": test_metrics},
                f,
                indent=2,
                ensure_ascii=False,
            )

    print("Done.")


if __name__ == "__main__":
    main()
