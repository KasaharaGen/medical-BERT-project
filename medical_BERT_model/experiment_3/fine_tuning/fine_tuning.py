#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fine_tuning_distill_optuna_ddp.py

要件:
- Phase2のbest_model/tokenizerをベースにカルテ2値分類
- 単純FTではなく蒸留(KD)を伴うFT
- 可能ならLoRA(ユーザ要望のrolaをLoRA想定)をstudentへ適用
- OptunaでHPO
- DDP + Optuna (rank0のみがStudy管理し、trial paramsを全rankへbroadcast)
- trialごとのbest_model管理: study_dir/trial_00001/... に保存
  best trialは study_dir/best_trial/ に同期

起動例(4GPU):
  torchrun --nproc_per_node=4 fine_tuning_distill_optuna_ddp.py \
    --student_model_dir ../pretraining_bert_2/pretraining_bert_best/best_model \
    --teacher_model_dir ../pretraining_bert_2/pretraining_bert_best/best_model \
    --tokenizer_dir     ../pretraining_bert_2/pretraining_bert_best/tokenizer \
    --csv               ../data/learning_data.csv \
    --output_dir        ./result_distill \
    --use_optuna \
    --study_dir         ./result_distill/optuna_study \
    --n_trials          30 \
    --final_train_best

注意:
- peft未導入ならLoRAは警告してスキップ(通常学習)
- Optunaのpruningは「rank0判定→全rankへ停止フラグbroadcast→Trainerにstop」方式
- test setはHPO中に使わない(best params確定後のfinal trainのみtest評価)
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
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.distributed as dist

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
from transformers.trainer_callback import TrainerCallback
from transformers.trainer import unwrap_model

import matplotlib.pyplot as plt


# =========================================================
# 基本ユーティリティ
# =========================================================
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_rank0() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def bcast_object(obj: Any, src: int = 0) -> Any:
    """
    rank0->全rankへPython objectをbroadcast
    """
    if not is_dist_avail_and_initialized():
        return obj
    obj_list = [obj] if is_rank0() else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def allreduce_float(value: float) -> float:
    """
    全rankで同一値を想定するが、念のため平均して返す
    """
    if not is_dist_avail_and_initialized():
        return float(value)
    t = torch.tensor([value], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / float(get_world_size())
    return float(t.item())


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


# =========================================================
# データ
# =========================================================
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


# =========================================================
# 指標
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "cm_00": cm[0][0],
        "cm_01": cm[0][1],
        "cm_10": cm[1][0],
        "cm_11": cm[1][1],
    }


def build_class_weight(train_labels: np.ndarray) -> Optional[torch.Tensor]:
    pos = int(train_labels.sum())
    neg = len(train_labels) - pos
    if pos == 0 or neg == 0:
        return None
    w0 = len(train_labels) / (2.0 * neg)
    w1 = len(train_labels) / (2.0 * pos)
    return torch.tensor([w0, w1], dtype=torch.float)


# =========================================================
# 可視化・履歴保存
# =========================================================
def save_history_and_plots(trainer: Trainer, out_dir: str) -> None:
    if not trainer.is_world_process_zero():
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Loss
    if "loss" in df.columns or "eval_loss" in df.columns:
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


def plot_confusion_matrix_matplotlib(cm: np.ndarray, out_path: str, title: str) -> None:
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


# =========================================================
# LoRA (任意)
# =========================================================
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


# =========================================================
# 蒸留Trainer
# =========================================================
class DistillTrainer(Trainer):
    """
    蒸留付き損失:
      (1-alpha)*CE + alpha*T^2*KL( teacher||student )
    CE側にのみ class_weight / label_smoothing を適用
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        distill_alpha: float = 0.5,
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # student forward
        outputs_s = model(**model_inputs)
        logits_s = outputs_s.logits

        # ★ここが重要：teacherをstudentと同じdeviceへ
        teacher_device = next(self.teacher_model.parameters()).device
        student_device = logits_s.device
        if teacher_device != student_device:
            self.teacher_model.to(student_device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # teacher forward (no grad)
        with torch.no_grad():
            outputs_t = self.teacher_model(**model_inputs)
            logits_t = outputs_t.logits

        # CE loss
        weight = self.class_weight.to(student_device) if self.class_weight is not None else None
        ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss_ce = ce(logits_s.view(-1, logits_s.size(-1)), labels.view(-1))

        # KD loss
        T = self.temperature
        log_p_s = torch.log_softmax(logits_s / T, dim=-1)
        p_t = torch.softmax(logits_t / T, dim=-1)
        loss_kd = self.kl(log_p_s, p_t) * (T * T)

        alpha = self.distill_alpha
        loss = (1.0 - alpha) * loss_ce + alpha * loss_kd

        return (loss, outputs_s) if return_outputs else loss


# =========================================================
# Optuna pruning を DDP全rankで安全に止めるCallback
# =========================================================
class DDPOptunaPruneCallback(TrainerCallback):
    """
    evalのたびにrank0がtrial.report/should_pruneを判断し、
    pruneするなら全rankへフラグbroadcastして training_stop させる。
    """
    def __init__(self, trial: optuna.Trial, monitor: str = "eval_mcc"):
        self.trial = trial
        self.monitor = monitor
        self._prune_now = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        # monitor値が無いなら何もしない
        val = metrics.get(self.monitor)
        if val is None:
            return control

        # rank0のみがoptunaにreport
        if is_rank0():
            self.trial.report(float(val), step=int(state.global_step))
            if self.trial.should_prune():
                self._prune_now = True

        # pruneフラグを全rankへ同期
        self._prune_now = bcast_object(self._prune_now, src=0)

        if self._prune_now:
            control.should_training_stop = True
        return control

    @property
    def pruned(self) -> bool:
        return bool(self._prune_now)


# =========================================================
# テスト評価(MCC最大閾値探索) - final trainのみで使用
# =========================================================
@torch.no_grad()
def distributed_test_eval_and_save(
    model,
    test_ds,
    tokenizer,
    output_dir: str,
    per_device_batch_size: int,
    amp_dtype: Optional[str],
):
    # HF TrainerのDDP下でdistは初期化済みの想定
    is_dist = is_dist_avail_and_initialized()
    world_size = get_world_size()
    rank = get_rank()

    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(test_ds, shuffle=False) if is_dist else None
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        test_ds,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    device = f"cuda:{rank}" if torch.cuda.is_available() and is_dist else ("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = unwrap_model(model).to(device)
    base_model.eval()

    use_amp = torch.cuda.is_available() and (amp_dtype in ["fp16", "bf16"])
    if use_amp and amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    elif use_amp and amp_dtype == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    all_labels_local, all_scores_local = [], []
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_labels_local.append(labels)
            all_scores_local.append(probs.detach().cpu().numpy())

    labels_local = np.concatenate(all_labels_local) if len(all_labels_local) else np.array([], dtype=np.int64)
    scores_local = np.concatenate(all_scores_local) if len(all_scores_local) else np.array([], dtype=np.float32)

    if is_dist:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (labels_local, scores_local))
        if rank != 0:
            return
        labels_all = np.concatenate([g[0] for g in gathered if g[0].size > 0])
        scores_all = np.concatenate([g[1] for g in gathered if g[1].size > 0])
    else:
        labels_all, scores_all = labels_local, scores_local

    if labels_all.size == 0:
        if is_rank0():
            print("[rank0] Empty test set. skip.")
        return

    preds_05 = (scores_all >= 0.5).astype(int)
    cm_05 = confusion_matrix(labels_all, preds_05, labels=[0, 1])
    mcc_05 = matthews_corrcoef(labels_all, preds_05) if len(np.unique(preds_05)) > 1 else 0.0

    thresholds = np.linspace(0.0, 1.0, 201)
    best_thr, best_mcc, best_cm = 0.5, -1.0, None
    for thr in thresholds:
        preds = (scores_all >= thr).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        mcc = matthews_corrcoef(labels_all, preds)
        if mcc > best_mcc:
            best_mcc, best_thr = float(mcc), float(thr)
            best_cm = confusion_matrix(labels_all, preds, labels=[0, 1])

    if best_cm is None:
        best_cm, best_mcc, best_thr = cm_05, float(mcc_05), 0.5

    tn, fp, fn, tp = best_cm.ravel()
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    test_metrics = {
        "world_size": int(world_size),
        "eval_accuracy": float(acc),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "eval_mcc": float(best_mcc),
        "eval_cm_00": int(tn), "eval_cm_01": int(fp),
        "eval_cm_10": int(fn), "eval_cm_11": int(tp),
        "best_threshold": float(best_thr),
        "eval_mcc_thr_0_5": float(mcc_05),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    cm_df = pd.DataFrame(best_cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(os.path.join(output_dir, "test_confusion_matrix.csv"))

    plot_confusion_matrix_matplotlib(
        best_cm,
        out_path=os.path.join(output_dir, "test_confusion_matrix.png"),
        title=f"Test Confusion Matrix (best thr={best_thr:.2f})",
    )

    print("[rank0] aggregated test metrics saved.")


# =========================================================
# 1 trial の学習(=train+eval)を実行して eval_mcc を返す
#  - HPO中は test を使わない
#  - 保存先は trial_dir に統一
# =========================================================
def run_one_trial_train_eval(
    args,
    trial_params: Dict[str, Any],
    trial_dir: str,
    use_prune: bool,
    optuna_trial: Optional[optuna.Trial] = None,
) -> Dict[str, float]:
    """
    returns: eval metrics dict (rank0 metricsを想定)
    """
    # dirs
    if is_rank0():
        os.makedirs(trial_dir, exist_ok=True)
    barrier()

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)

    # dataset
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

    # amp
    amp_dtype = None
    if args.amp == "fp16":
        amp_dtype = "fp16"
    elif args.amp == "bf16":
        amp_dtype = "bf16"

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

    # models
    student = AutoModelForSequenceClassification.from_pretrained(args.student_model_dir, config=student_config)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_dir, config=teacher_config)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
    if use_lora and (not lora_on) and is_rank0():
        print("[WARN] peft未導入のため LoRA をスキップした")

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    # TrainingArguments (trialごとにoutput_dirを変える)
    tr_args = TrainingArguments(
        output_dir=trial_dir,
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
        save_strategy="no",
        save_steps=int(args.eval_steps),
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="mcc",
        greater_is_better=True,
        seed=int(args.seed),
        report_to=["none"],
        dataloader_num_workers=4,
        fp16=(amp_dtype == "fp16"),
        bf16=(amp_dtype == "bf16"),
        disable_tqdm=not is_rank0(),
    )

    # distill params
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
        compute_metrics=compute_metrics,
        teacher_model=teacher,
        distill_alpha=distill_alpha,
        temperature=temperature,
        class_weight=class_weight,
        label_smoothing=label_smoothing,
    )

    prune_cb = None
    if use_prune and (optuna_trial is not None):
        prune_cb = DDPOptunaPruneCallback(optuna_trial, monitor="eval_mcc")
        trainer.add_callback(prune_cb)

    # train
    trainer.train()

    # evaluate
    metrics = trainer.evaluate()

    # 念のため全rankで同一値へ寄せる(平均)
    if "eval_mcc" in metrics:
        metrics["eval_mcc"] = allreduce_float(float(metrics["eval_mcc"]))

    # artifacts保存（rank0のみ）
    if trainer.is_world_process_zero():
        # best_model/tokenizer を trial_dir に保存
        best_model_dir = os.path.join(trial_dir, "best_model")
        tok_dir = os.path.join(trial_dir, "tokenizer")
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(tok_dir, exist_ok=True)

        trainer.save_model(best_model_dir)
        tok.save_pretrained(tok_dir)

        save_history_and_plots(trainer, trial_dir)

        # trial設定保存
        cfg = dict(trial_params)
        cfg.update({
            "student_model_dir": args.student_model_dir,
            "teacher_model_dir": args.teacher_model_dir,
            "tokenizer_dir": args.tokenizer_dir,
            "use_lora_effective": bool(lora_on),
        })
        with open(os.path.join(trial_dir, "trial_params.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        with open(os.path.join(trial_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    barrier()
    return metrics


# =========================================================
# best_trial同期（rank0のみ）
# =========================================================
def sync_best_trial_artifacts(study_dir: str, trial_dir: str, score: float):
    """
    study_dir/best_trial を trial_dir で上書き同期する
    """
    if not is_rank0():
        return
    best_dir = os.path.join(study_dir, "best_trial")
    os.makedirs(best_dir, exist_ok=True)

    # 既存bestを一旦退避
    tmp_dir = os.path.join(study_dir, "_best_trial_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(best_dir):
        shutil.move(best_dir, tmp_dir)

    try:
        shutil.copytree(trial_dir, best_dir)
        meta = {"score_eval_mcc": float(score), "synced_from": os.path.basename(trial_dir)}
        with open(os.path.join(best_dir, "best_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        # 失敗したら元に戻す
        print(f"[WARN] best_trial sync failed: {e}")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir, ignore_errors=True)
        if os.path.exists(tmp_dir):
            shutil.move(tmp_dir, best_dir)


# =========================================================
# rank0制御でOptunaを回す
# =========================================================
def run_optuna_ddp(args):
    """
    rank0:
      - studyを作る
      - for trial in n_trials:
          paramsをsample
          broadcast
          全rankでDDP学習
          rank0がtell
          bestならbest_trial同期
    non-rank0:
      - rank0からparamsを受け取り、同じtrialをDDPで回す
    """
    study_dir = args.study_dir
    if is_rank0():
        os.makedirs(study_dir, exist_ok=True)

    barrier()

    storage = None
    if args.optuna_storage:
        # 例: sqlite:///.../study.db
        storage = args.optuna_storage

    # rank0のみStudy生成
    study = None
    if is_rank0():
        pruner = MedianPruner(n_startup_trials=max(5, args.n_startup_trials))
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
        )

    best_score = -1e9
    if is_rank0() and len(study.trials) > 0 and study.best_trial is not None:
        try:
            best_score = float(study.best_value)
        except Exception:
            pass

    # HPOループ
    for t in range(args.n_trials):
        # --- rank0: trialをaskしてparamsサンプル ---
        if is_rank0():
            trial = study.ask()
            trial_number = trial.number

            # 探索空間（必要に応じて追加）
            # 重要: LoRAは条件付きにしている
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-2),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
                "distill_alpha": trial.suggest_float("distill_alpha", 0.1, 0.9),
                "temperature": trial.suggest_float("temperature", 1.0, 4.0),
                "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
                "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.3),
                "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.3),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "grad_accum": trial.suggest_categorical("grad_accum", [1, 2, 4]),
                "epochs": trial.suggest_categorical("epochs", [2, 3, 4]),
                # LoRA
                "use_lora": True
            }

            if params["use_lora"]:
                params["lora_r"] = trial.suggest_categorical("lora_r", [4, 8, 16])
                params["lora_alpha"] = trial.suggest_categorical("lora_alpha", [8, 16, 32])
                params["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.1)
                params["lora_targets"] = args.lora_targets

            ctrl = {"trial_number": trial_number, "params": params, "stop": False}
        else:
            trial = None
            ctrl = None

        # --- 全rankへbroadcast ---
        ctrl = bcast_object(ctrl, src=0)
        if ctrl.get("stop", False):
            break

        trial_number = int(ctrl["trial_number"])
        params = dict(ctrl["params"])

        # trial_dir
        trial_dir = os.path.join(study_dir, f"trial_{trial_number:05d}")

        # --- 全rankで学習 ---
        # pruningを有効にする場合は optuna_trial を渡す必要があるが、
        # rank0のみが本物のtrialを持つため、
        # rank0: optuna_trial=trial / others: None
        use_prune = bool(args.use_pruner)
        optuna_trial = trial if is_rank0() else None

        metrics = run_one_trial_train_eval(
            args=args,
            trial_params=params,
            trial_dir=trial_dir,
            use_prune=use_prune,
            optuna_trial=optuna_trial,
        )

        # rank0でtell
        if is_rank0():
            score = float(metrics.get("eval_mcc", -1e9))
            pruned = False
            if use_prune and (optuna_trial is not None):
                # callbackがstopを立てた場合もあるので、metricsの存在とは独立に判定
                # stopで終わっていればpruned扱いにする（Optuna上）
                # ※厳密にはcallback状態を参照したいが簡単化
                pruned = False

            # stop理由がpruneであるかの判定は「eval_mccが極端に欠損」等もあり得るため、
            # trainer側でprune時は通常 eval が出る場合が多い。ここでは should_prune が効いたならPRUNEDに倒す。
            # optuna側は report/should_prune を trial内で回しているので、終了後に状態を決める。
            if use_prune and (optuna_trial is not None) and optuna_trial.should_prune():
                pruned = True

            if pruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f"[rank0] Trial {trial_number} pruned. eval_mcc={score:.5f}")
            else:
                study.tell(trial, score)
                print(f"[rank0] Trial {trial_number} finished. eval_mcc={score:.5f}")

                # best更新なら同期
                if score > best_score:
                    best_score = score
                    sync_best_trial_artifacts(study_dir, trial_dir, score)

        barrier()

    # HPO終了を全rankへ通知（形式上）
    if is_rank0():
        ctrl = {"stop": True}
    else:
        ctrl = None
    _ = bcast_object(ctrl, src=0)
    barrier()

    # best paramsを保存
    if is_rank0() and study is not None and study.best_trial is not None:
        best = {
            "best_value": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_params": dict(study.best_trial.params),
        }
        with open(os.path.join(study_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)

    barrier()
    return


# =========================================================
# best paramsで本学習→テスト
# =========================================================
def final_train_best(args):
    """
    study_dir/best_params.json を読み best paramsで再学習し、output_dirに成果物を保存し test評価まで行う
    """
    must_dir(args.study_dir, "STUDY_DIR")
    best_path = os.path.join(args.study_dir, "best_params.json")
    must_file(best_path, "best_params.json")

    if is_rank0():
        best = json.loads(Path(best_path).read_text(encoding="utf-8"))
        best_params = best["best_params"]
        best_trial_number = best["best_trial_number"]
    else:
        best_params = None
        best_trial_number = None

    best_params = bcast_object(best_params, src=0)
    best_trial_number = bcast_object(best_trial_number, src=0)

    # final出力先
    out_dir = args.output_dir
    if is_rank0():
        os.makedirs(out_dir, exist_ok=True)
    barrier()

    # final run
    metrics = run_one_trial_train_eval(
        args=args,
        trial_params=best_params,
        trial_dir=out_dir,
        use_prune=False,
        optuna_trial=None,
    )

    # test（finalのみ）
    tok = AutoTokenizer.from_pretrained(os.path.join(out_dir, "tokenizer"), use_fast=True)
    dsd = build_datasets_from_csv(
        csv_path=args.csv,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tokenizer=tok,
        max_length=int(best_params.get("max_length", args.max_length)),
    )

    # best_modelで読み直し（rank0保存済みを想定）
    barrier()
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(out_dir, "best_model"))
    amp_dtype = None
    if args.amp == "fp16":
        amp_dtype = "fp16"
    elif args.amp == "bf16":
        amp_dtype = "bf16"

    distributed_test_eval_and_save(
        model=model,
        test_ds=dsd["test"],
        tokenizer=tok,
        output_dir=out_dir,
        per_device_batch_size=int(best_params.get("batch_size", args.batch_size)),
        amp_dtype=amp_dtype,
    )

    if is_rank0():
        meta = {
            "best_trial_number": int(best_trial_number),
            "best_params_used": dict(best_params),
            "final_eval_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        with open(os.path.join(out_dir, "final_train_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    barrier()
    return


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="KD fine-tuning + Optuna + DDP(rank0 control) + trial artifact management.")

    # モデル
    p.add_argument("--student_model_dir", type=str, required=True)
    p.add_argument("--teacher_model_dir", type=str, required=True)
    p.add_argument("--tokenizer_dir", type=str, required=True)

    # データ
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)

    # 出力
    p.add_argument("--output_dir", type=str, default="./result_distill")

    # 学習（デフォルト）
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
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # KD
    p.add_argument("--distill_alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)

    # AMP
    p.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])

    # logging/eval
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=50)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="query,key,value,dense")

    # Optuna
    p.add_argument("--use_optuna", action="store_true", default=False)
    p.add_argument("--study_dir", type=str, default="./optuna_study")
    p.add_argument("--study_name", type=str, default="kd_seqcls")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_startup_trials", type=int, default=5)
    p.add_argument("--use_pruner", action="store_true", default=True)
    p.add_argument("--optuna_storage", type=str, default="", help="例: sqlite:////path/to/study.db (空ならin-memory)")
    p.add_argument("--allow_lora_search", action="store_true", default=True)

    # bestで本学習
    p.add_argument("--final_train_best", action="store_true", default=True)

    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    # existence checks (rank0だけでよいが、エラーを早く出すため全rankで実施)
    must_dir(args.student_model_dir, "STUDENT_MODEL_DIR")
    must_dir(args.teacher_model_dir, "TEACHER_MODEL_DIR")
    must_dir(args.tokenizer_dir, "TOKENIZER_DIR")
    must_file(args.csv, "CSV")

    if args.amp == "none":
        args.amp = "none"

    # Optuna storage
    if args.optuna_storage == "":
        args.optuna_storage = None

    if args.use_optuna:
        run_optuna_ddp(args)
        if args.final_train_best:
            final_train_best(args)
    else:
        # Optuna無し: 1回だけ (output_dirに保存 + test評価)
        trial_params = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "distill_alpha": args.distill_alpha,
            "temperature": args.temperature,
            "label_smoothing": args.label_smoothing,
            "hidden_dropout_prob": args.hidden_dropout_prob,
            "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
            "classifier_dropout": args.classifier_dropout,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "use_lora": bool(args.use_lora),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
        }
        _ = run_one_trial_train_eval(
            args=args,
            trial_params=trial_params,
            trial_dir=args.output_dir,
            use_prune=False,
            optuna_trial=None,
        )

        # test
        tok = AutoTokenizer.from_pretrained(os.path.join(args.output_dir, "tokenizer"), use_fast=True)
        dsd = build_datasets_from_csv(
            csv_path=args.csv,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            tokenizer=tok,
            max_length=int(args.max_length),
        )

        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "best_model"))
        amp_dtype = None
        if args.amp == "fp16":
            amp_dtype = "fp16"
        elif args.amp == "bf16":
            amp_dtype = "bf16"

        distributed_test_eval_and_save(
            model=model,
            test_ds=dsd["test"],
            tokenizer=tok,
            output_dir=args.output_dir,
            per_device_batch_size=int(args.batch_size),
            amp_dtype=amp_dtype,
        )

    if is_rank0():
        print("Done.")


if __name__ == "__main__":
    main()
