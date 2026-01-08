#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fine_tuning_single_gpu_kd_lora_optuna.py

目的:
- 事前学習+ドメイン学習済みEncoderをカルテ2値分類に適応
- 単なるFTではなく Knowledge Distillation(KD) を伴うFT
- 可能ならLoRA(PEFT)を student に適用
- OptunaでHPO（validationは閾値最適化MCCを採用：最優先）
- GPU1枚（DDPなし）

最優先改善:
- validationでも確率→閾値走査でbest MCCを計算し、Optuna目的にする

高優先改善:
- distill_alpha探索範囲を0.2〜0.5に制限（FP過多抑制）
- LoRA使用時でも classifier は full update（境界調整能力を確保）

実行例:
python fine_tuning_single_gpu_kd_lora_optuna.py \
  --student_model_dir ../pretraining_bert_2/pretraining_bert_best/best_model \
  --teacher_model_dir ../pretraining_bert_2/pretraining_bert_best/best_model \
  --tokenizer_dir     ../pretraining_bert_2/pretraining_bert_best/tokenizer \
  --csv               ../data/learning_data.csv \
  --output_dir        ./result_distill_single \
  --study_dir         ./result_distill_single/optuna_study \
  --n_trials          20 \
  --use_optuna \
  --use_lora \
  --final_train_best
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import time
import random
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from torch import nn

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

import optuna
from optuna.pruners import MedianPruner

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
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


# =========================
# データ
# =========================
def stratified_three_split(
    df: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ({"text", "label"} <= set(df.columns)):
        raise RuntimeError("CSVに text,label 列が必要である。")
    train_rest, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["label"], shuffle=True
    )
    rel_val_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_rest,
        test_size=rel_val_ratio,
        random_state=seed,
        stratify=train_rest["label"],
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_datasets_from_csv(
    csv_path: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    df = pd.read_csv(csv_path)
    tr, va, te = stratified_three_split(df[["text", "label"]], test_ratio, val_ratio, seed)

    ds_tr = Dataset.from_pandas(tr)
    ds_va = Dataset.from_pandas(va)
    ds_te = Dataset.from_pandas(te)

    def _tok_map(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    def _prep(ds):
        ds = ds.map(_tok_map, batched=True, remove_columns=["text"])
        ds = ds.rename_column("label", "labels")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    return DatasetDict({"train": _prep(ds_tr), "validation": _prep(ds_va), "test": _prep(ds_te)})


def build_class_weight(train_labels: np.ndarray) -> Optional[torch.Tensor]:
    pos = int(train_labels.sum())
    neg = len(train_labels) - pos
    if pos == 0 or neg == 0:
        return None
    w0 = len(train_labels) / (2.0 * neg)
    w1 = len(train_labels) / (2.0 * pos)
    return torch.tensor([w0, w1], dtype=torch.float)


# =========================
# 最優先: validationで閾値最適化MCC
# =========================
def metrics_threshold_sweep_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    n_grid: int = 201,
) -> Dict[str, float]:
    """
    logits -> softmax(prob1) -> threshold sweep -> best MCC
    """
    # 安定のためsoftmax
    logits_t = torch.tensor(logits, dtype=torch.float32)
    prob1 = torch.softmax(logits_t, dim=-1)[:, 1].cpu().numpy()

    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_thr = 0.5
    best_mcc = -1.0
    best_f1 = 0.0
    best_acc = 0.0
    best_cm = None

    for thr in thresholds:
        preds = (prob1 >= thr).astype(int)
        if len(np.unique(preds)) < 2:
            # MCCが定義できない（全予測同一）ケースをスキップ
            continue
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_thr = float(thr)
            best_f1 = float(f1_score(labels, preds))
            best_acc = float(accuracy_score(labels, preds))
            best_cm = confusion_matrix(labels, preds, labels=[0, 1])

    # 全予測同一しか出なかった場合のフォールバック
    if best_cm is None:
        preds = (prob1 >= 0.5).astype(int)
        best_thr = 0.5
        best_f1 = float(f1_score(labels, preds)) if len(np.unique(preds)) > 1 else 0.0
        best_acc = float(accuracy_score(labels, preds))
        best_mcc = float(matthews_corrcoef(labels, preds)) if len(np.unique(preds)) > 1 else 0.0
        best_cm = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = best_cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "mcc": float(best_mcc),
        "f1": float(best_f1),
        "accuracy": float(best_acc),
        "precision": float(precision),
        "recall": float(recall),
        "best_threshold": float(best_thr),
        "cm_00": int(tn), "cm_01": int(fp),
        "cm_10": int(fn), "cm_11": int(tp),
    }


def compute_metrics_threshold_mcc(eval_pred):
    logits, labels = eval_pred
    out = metrics_threshold_sweep_from_logits(logits, labels, n_grid=201)
    # Trainerは "eval_" prefix を付ける。ここではkeysを素直に返す。
    return {
        "mcc": out["mcc"],
        "f1": out["f1"],
        "accuracy": out["accuracy"],
        "precision": out["precision"],
        "recall": out["recall"],
        "best_threshold": out["best_threshold"],
        "cm_00": out["cm_00"], "cm_01": out["cm_01"],
        "cm_10": out["cm_10"], "cm_11": out["cm_11"],
    }


# =========================
# LoRA (任意)
# =========================
def maybe_apply_lora_to_student(
    model: torch.nn.Module,
    enabled: bool,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Tuple[str, ...],
) -> Tuple[torch.nn.Module, bool]:
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


def force_classifier_trainable(model: torch.nn.Module) -> None:
    """
    高優先: LoRAを使ってもclassifierはfull updateする
    PEFT適用後はbaseがfreezeされがちなので、classifierを明示的にtrainableにする。
    """
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True


# =========================
# 蒸留Trainer
# =========================
class DistillTrainer(Trainer):
    """
    蒸留付き損失:
      (1-alpha)*CE + alpha*T^2*KL( teacher||student )

    CE側にのみ class_weight / label_smoothing を適用
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        distill_alpha: float = 0.35,
        temperature: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_alpha = float(distill_alpha)
        self.temperature = float(temperature)
        self.class_weight = class_weight
        self.label_smoothing = float(label_smoothing)
        self.kl = nn.KLDivLoss(reduction="batchmean")

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # student forward
        out_s = model(**model_inputs)
        logits_s = out_s.logits
        student_device = logits_s.device

        # teacherをstudentと同deviceへ（device mismatch回避）
        t_dev = next(self.teacher_model.parameters()).device
        if t_dev != student_device:
            self.teacher_model.to(student_device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # teacher forward
        with torch.no_grad():
            out_t = self.teacher_model(**model_inputs)
            logits_t = out_t.logits

        # CE
        weight = self.class_weight.to(student_device) if self.class_weight is not None else None
        ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss_ce = ce(logits_s.view(-1, logits_s.size(-1)), labels.view(-1))

        # KD
        T = self.temperature
        log_p_s = torch.log_softmax(logits_s / T, dim=-1)
        p_t = torch.softmax(logits_t / T, dim=-1)
        loss_kd = self.kl(log_p_s, p_t) * (T * T)

        a = self.distill_alpha
        loss = (1.0 - a) * loss_ce + a * loss_kd
        return (loss, out_s) if return_outputs else loss


# =========================
# 可視化
# =========================
def save_history_and_plots(trainer: Trainer, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Loss
    if ("loss" in df.columns) or ("eval_loss" in df.columns):
        df_tr = df[df.get("loss").notna()] if "loss" in df.columns else pd.DataFrame()
        df_ev = df[df.get("eval_loss").notna()] if "eval_loss" in df.columns else pd.DataFrame()

        if len(df_tr) > 0 or len(df_ev) > 0:
            plt.figure()
            if len(df_tr) > 0:
                d = df_tr[["step", "loss"]].drop_duplicates(subset="step")
                plt.plot(d["step"], d["loss"], label="train_loss")
            if len(df_ev) > 0:
                d = df_ev[["step", "eval_loss"]].drop_duplicates(subset="step")
                plt.plot(d["step"], d["eval_loss"], label="eval_loss")
            plt.xlabel("global_step")
            plt.ylabel("loss")
            plt.title("Loss (Train & Eval)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "curve_loss.png"), dpi=150)
            plt.close()

    # Metrics
    metric_keys = ["eval_mcc", "eval_f1", "eval_accuracy"]
    keys = [k for k in metric_keys if k in df.columns and df[k].notna().any()]
    if len(keys) > 0:
        plt.figure()
        for k in keys:
            d = df[df[k].notna()][["step", k]].drop_duplicates(subset="step")
            plt.plot(d["step"], d[k], label=k)
        plt.xlabel("global_step")
        plt.ylabel("score")
        plt.title("Eval Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_eval_metrics.png"), dpi=150)
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred_0", "Pred_1"])
    plt.yticks(tick_marks, ["True_0", "True_1"])

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# 1 trial 学習→validation評価
# =========================
def run_train_eval(
    args,
    trial_params: Dict[str, Any],
    output_dir: str,
    save_model: bool,
) -> Dict[str, Any]:
    """
    save_model=False: Optuna探索中（保存なし、ENOSPC回避）
    save_model=True : final train（best_model/tokenizer保存、曲線保存）
    """
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    dsd = build_datasets_from_csv(
        csv_path=args.csv,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tokenizer=tok,
        max_length=int(trial_params.get("max_length", args.max_length)),
    )

    # class weight
    labels_np = np.array(dsd["train"]["labels"])
    class_weight = build_class_weight(labels_np) if args.use_class_weight else None

    # configs
    student_config = AutoConfig.from_pretrained(
        args.student_model_dir,
        num_labels=2,
        hidden_dropout_prob=float(trial_params.get("hidden_dropout_prob", args.hidden_dropout_prob)),
        attention_probs_dropout_prob=float(trial_params.get("attention_probs_dropout_prob", args.attention_probs_dropout_prob)),
        classifier_dropout=float(trial_params.get("classifier_dropout", args.classifier_dropout)),
    )
    teacher_config = AutoConfig.from_pretrained(
        args.teacher_model_dir,
        num_labels=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
    )

    student = AutoModelForSequenceClassification.from_pretrained(args.student_model_dir, config=student_config)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_dir, config=teacher_config)

    # LoRA
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

    # 高優先: classifierはfull update
    force_classifier_trainable(student)

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    amp = args.amp
    fp16 = (amp == "fp16")
    bf16 = (amp == "bf16")

    # Optuna探索中は保存を切る（ENOSPC対策）
    if save_model:
        save_strategy = "steps"
        load_best = True
        save_steps = int(args.eval_steps)
        save_total_limit = 1
        report_to = ["none"]
    else:
        save_strategy = "no"
        load_best = False
        save_steps = None
        save_total_limit = 0
        report_to = ["none"]

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
        metric_for_best_model="mcc",
        greater_is_better=True,
        seed=int(args.seed),
        report_to=report_to,
        dataloader_num_workers=4,
        fp16=fp16,
        bf16=bf16,
        disable_tqdm=False,
    )

    # distill params（高優先: alphaは小さめが効きやすい）
    distill_alpha = float(trial_params.get("distill_alpha", args.distill_alpha))
    temperature = float(trial_params.get("temperature", args.temperature))
    label_smoothing = float(trial_params.get("label_smoothing", args.label_smoothing))

    trainer = DistillTrainer(
        model=student,
        args=tr_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics_threshold_mcc,  # 最優先
        teacher_model=teacher,
        distill_alpha=distill_alpha,
        temperature=temperature,
        class_weight=class_weight,
        label_smoothing=label_smoothing,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # 返り値は python型に寄せる
    metrics_out = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in metrics.items()}

    if save_model:
        os.makedirs(output_dir, exist_ok=True)
        best_model_dir = os.path.join(output_dir, "best_model")
        tok_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(tok_dir, exist_ok=True)

        trainer.save_model(best_model_dir)
        tok.save_pretrained(tok_dir)

        save_history_and_plots(trainer, output_dir)

        # 設定保存
        cfg = dict(trial_params)
        cfg.update({
            "student_model_dir": args.student_model_dir,
            "teacher_model_dir": args.teacher_model_dir,
            "tokenizer_dir": args.tokenizer_dir,
            "use_lora_effective": bool(lora_on),
        })
        with open(os.path.join(output_dir, "train_params.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    return metrics_out


# =========================
# final: test評価（閾値最適化 + CM保存）
# =========================
@torch.no_grad()
def eval_on_test_and_save(args, output_dir: str) -> None:
    tok = AutoTokenizer.from_pretrained(os.path.join(output_dir, "tokenizer"), use_fast=True)
    dsd = build_datasets_from_csv(
        csv_path=args.csv,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tokenizer=tok,
        max_length=args.max_length,
    )
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(output_dir, "best_model"))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    from torch.utils.data import DataLoader
    loader = DataLoader(dsd["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    all_logits = []
    all_labels = []
    amp = args.amp
    use_amp = torch.cuda.is_available() and (amp in ["fp16", "bf16"])
    autocast_dtype = torch.float16 if amp == "fp16" else (torch.bfloat16 if amp == "bf16" else None)

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    out = metrics_threshold_sweep_from_logits(logits, labels, n_grid=201)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    cm = np.array([[out["cm_00"], out["cm_01"]], [out["cm_10"], out["cm_11"]]], dtype=int)
    pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"]).to_csv(
        os.path.join(output_dir, "test_confusion_matrix.csv")
    )
    plot_confusion_matrix(
        cm,
        out_path=os.path.join(output_dir, "test_confusion_matrix.png"),
        title=f"Test Confusion Matrix (best thr={out['best_threshold']:.2f})",
    )


# =========================
# Optuna
# =========================
def run_optuna(args) -> Dict[str, Any]:
    """
    ENOSPC回避のため、探索中はモデル保存しない。
    trialごとに params と eval_mcc だけを study_dir にCSV/JSONで残す。
    """
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
        pd.DataFrame(columns=["trial_number", "eval_mcc", "eval_f1", "eval_accuracy", "best_threshold", "params_json"]).to_csv(
            trials_log_path, index=False
        )

    def objective(trial: optuna.Trial) -> float:
        # 高優先: alpha/T範囲を絞る
        params = {
            # 最重要
            "lr": trial.suggest_float("lr", 5e-6, 3e-4, log=True),
            "distill_alpha": trial.suggest_float("distill_alpha", 0.2, 0.5),
            "temperature": trial.suggest_float("temperature", 1.5, 3.0),

            # 次点
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 5e-2),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),

            "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.2),
            "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.4),

            # 実効バッチ
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "grad_accum": trial.suggest_categorical("grad_accum", [1, 2, 4, 8]),

            # 固定（探索を広げない）
            "epochs": args.epochs,
            "max_length": args.max_length,
            "lr_scheduler_type": args.lr_scheduler_type,

            # LoRA（使う方針なら固定ON推奨）
            "use_lora": bool(args.use_lora),
            "lora_targets": args.lora_targets,
        }

        if params["use_lora"]:
            params["lora_r"] = trial.suggest_categorical("lora_r", [4, 8, 16])
            # alphaはrに合わせて絞る（無駄探索削減）
            if params["lora_r"] == 4:
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [8, 16])
            elif params["lora_r"] == 8:
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [16, 32])
            else:
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [32, 64])
            params["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.1)

        # trial_dirは作らない（保存しない）。output_dirは使い捨てでOKだが、Trainerはoutput_dir必須なので一時dirにする
        tmp_dir = os.path.join(args.study_dir, "_tmp_run")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        metrics = run_train_eval(args, params, output_dir=tmp_dir, save_model=False)
        eval_mcc = float(metrics.get("eval_mcc", metrics.get("mcc", -1e9)))
        # transformersは "eval_mcc" の形で返るが、compute_metricsは "mcc" で返すので最終的に eval_mcc になる
        if "eval_mcc" not in metrics and "mcc" in metrics:
            eval_mcc = float(metrics["mcc"])

        # pruning report
        trial.report(eval_mcc, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # log追記（最小情報のみ）
        row = {
            "trial_number": int(trial.number),
            "eval_mcc": float(eval_mcc),
            "eval_f1": float(metrics.get("eval_f1", metrics.get("f1", 0.0))),
            "eval_accuracy": float(metrics.get("eval_accuracy", metrics.get("accuracy", 0.0))),
            "best_threshold": float(metrics.get("eval_best_threshold", metrics.get("best_threshold", 0.5))),
            "params_json": json.dumps(params, ensure_ascii=False),
        }
        df = pd.read_csv(trials_log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(trials_log_path, index=False)

        # tmp掃除
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return eval_mcc

    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": dict(study.best_trial.params),
    }
    # best_params には optunaが知っているパラメータだけが入るので、固定値・lora_targets等を足す
    # distill/opt周りはbest_paramsで上書きし、残りはargsから補う
    merged = {
        "lr": best["best_params"].get("lr", args.lr),
        "weight_decay": best["best_params"].get("weight_decay", args.weight_decay),
        "warmup_ratio": best["best_params"].get("warmup_ratio", args.warmup_ratio),
        "label_smoothing": best["best_params"].get("label_smoothing", args.label_smoothing),

        "hidden_dropout_prob": best["best_params"].get("hidden_dropout_prob", args.hidden_dropout_prob),
        "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
        "classifier_dropout": best["best_params"].get("classifier_dropout", args.classifier_dropout),

        "batch_size": best["best_params"].get("batch_size", args.batch_size),
        "grad_accum": best["best_params"].get("grad_accum", args.grad_accum),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "lr_scheduler_type": args.lr_scheduler_type,

        "distill_alpha": best["best_params"].get("distill_alpha", args.distill_alpha),
        "temperature": best["best_params"].get("temperature", args.temperature),

        "use_lora": bool(args.use_lora),
        "lora_targets": args.lora_targets,
    }

    if args.use_lora:
        merged["lora_r"] = best["best_params"].get("lora_r", args.lora_r)
        merged["lora_alpha"] = best["best_params"].get("lora_alpha", args.lora_alpha)
        merged["lora_dropout"] = best["best_params"].get("lora_dropout", args.lora_dropout)

    out = {"best": best, "merged_best_params": merged}
    with open(os.path.join(args.study_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--student_model_dir", type=str, required=True)
    p.add_argument("--teacher_model_dir", type=str, required=True)
    p.add_argument("--tokenizer_dir", type=str, required=True)

    # data
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)

    # output
    p.add_argument("--output_dir", type=str, default="./result_distill_single")
    p.add_argument("--study_dir", type=str, default="./optuna_study")

    # train defaults
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=3e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine"])

    # dropout
    p.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    p.add_argument("--attention_probs_dropout_prob", type=float, default=0.0)
    p.add_argument("--classifier_dropout", type=float, default=0.1)

    # loss
    p.add_argument("--use_class_weight", action="store_true", default=True)
    p.add_argument("--label_smoothing", type=float, default=0.05)

    # KD（高優先: alphaは小さめを基本値に）
    p.add_argument("--distill_alpha", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=2.0)

    # AMP
    p.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])

    # eval/log
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=50)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="query,key,value,dense")

    # optuna
    p.add_argument("--use_optuna", action="store_true", default=False)
    p.add_argument("--study_name", type=str, default="kd_seqcls_single")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_startup_trials", type=int, default=5)
    p.add_argument("--use_pruner", action="store_true", default=True)
    p.add_argument("--optuna_storage", type=str, default="", help="例: sqlite:////path/to/study.db（空ならin-memory）")

    # bestで本学習
    p.add_argument("--final_train_best", action="store_true", default=True)

    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    must_dir(args.student_model_dir, "STUDENT_MODEL_DIR")
    must_dir(args.teacher_model_dir, "TEACHER_MODEL_DIR")
    must_dir(args.tokenizer_dir, "TOKENIZER_DIR")
    must_file(args.csv, "CSV")

    if args.amp == "none":
        args.amp = "none"

    if args.optuna_storage == "":
        args.optuna_storage = None

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_optuna:
        best_info = run_optuna(args)
        merged = best_info["merged_best_params"]

        if args.final_train_best:
            # final train（保存あり）
            _ = run_train_eval(args, merged, output_dir=args.output_dir, save_model=True)
            # test評価
            eval_on_test_and_save(args, args.output_dir)
    else:
        # 1回だけ（保存あり）
        params = {
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
            "use_lora": bool(args.use_lora),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
        }
        _ = run_train_eval(args, params, output_dir=args.output_dir, save_model=True)
        eval_on_test_and_save(args, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
