#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refactored fine_tuning.py

目的:
- 事前学習済み(Phase2)の "best_model/" と "tokenizer/" からロード
- 学習→検証→最良チェックポイントの保存→テスト(MCC最大閾値探索)まで一気通貫
- 設定は基本引数化し、既定値はBERT-base/医療ドメイン小規模を想定
- Optunaは任意でON/OFF。DB保存・並列ワーカーに配慮
- 成果物構成は Phase1/2 と統一:
    OUTPUT_DIR/
      ├── best_model/      ← 分類ヘッド込みの最終モデル
      ├── tokenizer/       ← 分類用に用いたトークナイザ
      ├── history.csv      ← Trainerのlog_history
      ├── curve_loss.png / curve_eval_metrics.png
      ├── test_metrics.(json|csv) / test_confusion_matrix.(csv|png)
      └── best_params.json（Optuna有効時のみ）
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import random
import argparse
from pathlib import Path
from typing import Optional, Tuple
import shutil

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed,
    TrainerCallback,
)
from transformers.trainer import unwrap_model

import matplotlib.pyplot as plt
import seaborn as sns

# ====== Optuna (任意) ======
import optuna
from optuna.pruners import MedianPruner


# -----------------------------
# 基本ユーティリティ
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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


# -----------------------------
# データセット構築
# -----------------------------
def stratified_three_split(df: pd.DataFrame, test_ratio: float, val_ratio: float, seed: int
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert {"text", "label"} <= set(df.columns), "CSVに text,label 列が必要である。"
    train_rest, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["label"], shuffle=True
    )
    rel_val_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_rest, test_size=rel_val_ratio, random_state=seed, stratify=train_rest["label"], shuffle=True
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_datasets_from_csv(csv_path: str, val_ratio: float, test_ratio: float, seed: int,
                            tokenizer, max_length: int, fp16: bool) -> DatasetDict:
    df = pd.read_csv(csv_path)
    tr, va, te = stratified_three_split(df[["text", "label"]], test_ratio, val_ratio, seed)

    ds_tr = Dataset.from_pandas(tr); ds_va = Dataset.from_pandas(va); ds_te = Dataset.from_pandas(te)

    def _tok_map(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        return out

    def _prep(ds):
        ds = ds.map(_tok_map, batched=True, remove_columns=["text"])
        ds = ds.rename_column("label", "labels")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    return DatasetDict({"train": _prep(ds_tr), "validation": _prep(ds_va), "test": _prep(ds_te)})


# -----------------------------
# 評価指標
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()
    return {
        "accuracy": acc, "f1": f1, "mcc": mcc,
        "cm_00": cm[0][0], "cm_01": cm[0][1],
        "cm_10": cm[1][0], "cm_11": cm[1][1],
    }


# -----------------------------
# 損失（クラス重み＋label smoothing）
# -----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -----------------------------
# 可視化・履歴保存
# -----------------------------
def save_history_and_plots(trainer: Trainer, out_dir: str):
    if not trainer.is_world_process_zero():
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Loss
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
        plt.xlabel("global_step"); plt.ylabel("loss"); plt.title("Loss (Train & Eval)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_loss.png"), dpi=150); plt.close()

    # Metrics
    eval_keys = [k for k in ["eval_mcc", "eval_f1", "eval_accuracy"] if k in df.columns and df[k].notna().any()]
    if len(eval_keys) > 0:
        plt.figure()
        for k in eval_keys:
            d = df[df[k].notna()][["step", k]].drop_duplicates(subset="step")
            plt.plot(d["step"], d[k], label=k)
        plt.xlabel("global_step"); plt.ylabel("score"); plt.title("Eval Metrics (MCC / F1 / Accuracy)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_eval_metrics.png"), dpi=150); plt.close()


# -----------------------------
# テスト集計（全rank集約＋MCC最大閾値探索）
# -----------------------------
@torch.no_grad()
def distributed_test_eval_and_save(
    model,
    test_ds,
    tokenizer,
    output_dir: str,
    per_device_batch_size: int,
    use_fp16: bool,
):
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    sampler = DistributedSampler(test_ds, shuffle=False) if is_dist else None
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_fp16 else None)
    loader = DataLoader(
        test_ds, batch_size=per_device_batch_size, shuffle=False,
        sampler=sampler, collate_fn=collator, num_workers=4, pin_memory=True
    )

    device = f"cuda:{rank}" if torch.cuda.is_available() and is_dist else ("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = unwrap_model(model).to(device)
    base_model.eval()

    all_labels_local, all_scores_local = [], []
    amp_ctx = torch.cuda.amp.autocast(enabled=use_fp16 and torch.cuda.is_available())
    with amp_ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            probs = torch.softmax(base_model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)[:, 1]
            all_labels_local.append(labels)
            all_scores_local.append(probs.detach().cpu().numpy())

    if len(all_labels_local) == 0:
        if rank == 0:
            print("[rank0] Empty test set. No metrics saved.")
        return

    labels_local = np.concatenate(all_labels_local)
    scores_local = np.concatenate(all_scores_local)

    if is_dist:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (labels_local, scores_local))
        if rank == 0:
            ys = [g[0] for g in gathered]
            ss = [g[1] for g in gathered]
            labels_all = np.concatenate(ys)
            scores_all = np.concatenate(ss)
        else:
            return
    else:
        labels_all, scores_all = labels_local, scores_local

    # MCC最大化閾値探索
    preds_05 = (scores_all >= 0.5).astype(int)
    cm_05 = confusion_matrix(labels_all, preds_05, labels=[0, 1])
    mcc_05 = matthews_corrcoef(labels_all, preds_05)

    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr, best_mcc, best_cm = 0.5, -1.0, None
    for thr in thresholds:
        preds = (scores_all >= thr).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        mcc = matthews_corrcoef(labels_all, preds)
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
            best_cm = confusion_matrix(labels_all, preds, labels=[0, 1])
    if best_cm is None:
        best_cm, best_mcc, best_thr = cm_05, mcc_05, 0.5

    tn, fp, fn, tp = best_cm.ravel()
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    test_metrics = {
        "world_size": world_size,
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_mcc": best_mcc,
        "eval_cm_00": int(tn), "eval_cm_01": int(fp),
        "eval_cm_10": int(fn), "eval_cm_11": int(tp),
        "best_threshold": float(best_thr),
        "eval_mcc_thr_0_5": float(mcc_05),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    cm_df = pd.DataFrame(best_cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(os.path.join(output_dir, "test_confusion_matrix.csv"))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 14, "weight": "bold", "color": "black"})
    plt.title(f"Test Confusion Matrix (best thr={best_thr:.2f})", fontsize=14)
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_confusion_matrix.png"), dpi=200)
    plt.close()
    print("[rank0] aggregated test metrics with threshold search saved.")


# -----------------------------
# Optuna 目的関数
# -----------------------------
def build_class_weight(train_labels: np.ndarray) -> Optional[torch.Tensor]:
    pos = int(train_labels.sum())
    neg = len(train_labels) - pos
    if pos == 0 or neg == 0:
        return None
    w0 = len(train_labels) / (2.0 * neg)
    w1 = len(train_labels) / (2.0 * pos)
    return torch.tensor([w0, w1], dtype=torch.float)


def objective(trial: optuna.Trial, args):
    # 単GPU想定の探索
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    dsd = build_datasets_from_csv(
        csv_path=args.csv,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tokenizer=tok,
        max_length=args.max_length,
        fp16=args.fp16,
    )

    # 探索空間
    lr   = trial.suggest_float("learning_rate", 1e-5, 7e-5, log=True)
    wd   = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    warm = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    dr_hid = trial.suggest_float("hidden_dropout_prob", 0.0, 0.3)
    dr_att = trial.suggest_float("attention_probs_dropout_prob", 0.0, 0.3)
    dr_cls = trial.suggest_float("classifier_dropout", 0.0, 0.5)

    lbl_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)
    sched = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"])
    per_bs = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32])
    grad_acc = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])

    # モデル構築（Phase2 best_modelから）
    config = AutoConfig.from_pretrained(
        args.model_dir,
        num_labels=2,
        hidden_dropout_prob=dr_hid,
        attention_probs_dropout_prob=dr_att,
        classifier_dropout=dr_cls,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=config)

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8 if args.fp16 else None)
    trial_out = os.path.join(args.output_dir, f"trial_{trial.number:03d}")
    os.makedirs(trial_out, exist_ok=True)

    trainer_args = TrainingArguments(
        output_dir=trial_out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_bs,
        per_device_eval_batch_size=max(8, per_bs),
        learning_rate=lr,
        lr_scheduler_type=sched,
        warmup_ratio=warm,
        weight_decay=wd,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to=["none"],
        seed=args.seed,
        fp16=args.fp16,
        dataloader_num_workers=4,
        gradient_accumulation_steps=grad_acc,
        load_best_model_at_end=False,
        metric_for_best_model="mcc",
        greater_is_better=True,
    )

    # Optuna に中間値を報告
    class ReportCB(TrainerCallback):
        def on_evaluate(self, args_, state, control, metrics=None, **kwargs):
            if metrics and "eval_mcc" in metrics:
                trial.report(metrics["eval_mcc"], step=int(state.global_step))
                if trial.should_prune():
                    raise optuna.TrialPruned()

    class_weight = build_class_weight(np.array(dsd["train"]["labels"]))

    trainer = WeightedTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weight=class_weight,
        label_smoothing=lbl_smooth,
    )
    trainer.add_callback(ReportCB())

    trainer.train()
    metrics = trainer.evaluate()
    mcc = float(metrics.get("eval_mcc", 0.0))

    # 軽量成果物
    try:
        save_history_and_plots(trainer, trial_out)
        with open(os.path.join(trial_out, "val_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return mcc


# -----------------------------
# メイン（本学習→テスト）
# -----------------------------
def train_and_test(args, best_params: Optional[dict] = None):
    set_all_seeds(args.seed)
    must_dir(args.model_dir, "MODEL_DIR(best_model)")
    must_dir(args.tokenizer_dir, "TOKENIZER_DIR(tokenizer)")
    must_file(args.csv, "CSV_PATH")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    dsd = build_datasets_from_csv(
        csv_path=args.csv,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tokenizer=tok,
        max_length=args.max_length,
        fp16=args.fp16,
    )

    labels_np = np.array(dsd["train"]["labels"])
    class_weight = build_class_weight(labels_np)

    hp = dict(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        classifier_dropout=args.classifier_dropout,
        label_smoothing=args.label_smoothing,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )
    if best_params:
        hp.update(best_params)

    config = AutoConfig.from_pretrained(
        args.model_dir,
        num_labels=2,
        hidden_dropout_prob=hp["hidden_dropout_prob"],
        attention_probs_dropout_prob=hp["attention_probs_dropout_prob"],
        classifier_dropout=hp["classifier_dropout"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=config)
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8 if args.fp16 else None)

    os.makedirs(args.output_dir, exist_ok=True)

    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=max(8, hp["per_device_train_batch_size"]),
        learning_rate=hp["learning_rate"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        fp16=args.fp16,
        dataloader_num_workers=4,
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        seed=args.seed,
        report_to=["none"],
    )

    trainer = WeightedTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weight=class_weight,
        label_smoothing=hp["label_smoothing"],
    )

    # 学習→検証
    trainer.train()
    _ = trainer.evaluate()

    # ===== 保存（Phase1/2と同構造） =====
    if trainer.is_world_process_zero():
        # best_model/
        best_model_dir = os.path.join(args.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)

        # tokenizer/
        tok_dir = os.path.join(args.output_dir, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok.save_pretrained(tok_dir)

        # 曲線・履歴
        save_history_and_plots(trainer, args.output_dir)

        # （Optuna時のみ）best_trial成果物コピー
        if best_params and args.enable_tuning:
            best_trial_dir = os.path.join(args.output_dir, f"trial_{args.best_trial_num:03d}")
            best_art_dir = os.path.join(args.output_dir, "best_trial")
            os.makedirs(best_art_dir, exist_ok=True)
            for fn in ["history.csv", "curve_loss.png", "curve_eval_metrics.png", "val_metrics.json"]:
                src = os.path.join(best_trial_dir, fn)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(best_art_dir, fn))

    # ===== テスト（集約＋MCC最大閾値探索） =====
    distributed_test_eval_and_save(
        model=trainer.model,
        test_ds=dsd["test"],
        tokenizer=tok,
        output_dir=args.output_dir,
        per_device_batch_size=hp["per_device_train_batch_size"],
        use_fp16=args.fp16,
    )
    print("Done.")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning with Optuna (optional) and unified outputs.")
    # 入力
    p.add_argument("--model_dir",     type=str, default="../pretraining_bert_2/pretraining_bert_best/best_model",
                   help="Phase2のbest_modelディレクトリ")
    p.add_argument("--tokenizer_dir", type=str, default="../pretraining_bert_2/pretraining_bert_best/tokenizer",
                   help="Phase2のtokenizerディレクトリ")
    p.add_argument("--csv",           type=str, default="../data/learning_data.csv",
                   help="text,label 列を含むCSV")

    # 出力
    p.add_argument("--output_dir",    type=str, default="./result", help="成果物の出力先")

    # 学習基本
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--epochs",        type=int, default=5)
    p.add_argument("--max_length",    type=int, default=512)
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--grad_accum",    type=int, default=1)
    p.add_argument("--fp16",          action="store_true", default=True)

    # スケジューリング
    p.add_argument("--lr",            type=float, default=1e-5)
    p.add_argument("--weight_decay",  type=float, default=3e-3)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine"])

    # ドロップアウト・損失
    p.add_argument("--hidden_dropout_prob",         type=float, default=0.2)
    p.add_argument("--attention_probs_dropout_prob", type=float, default=0.0)
    p.add_argument("--classifier_dropout",           type=float, default=0.1)
    p.add_argument("--label_smoothing",              type=float, default=0.1)

    # データ分割
    p.add_argument("--val_ratio",  type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)

    # ログ
    p.add_argument("--eval_steps",    type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=50)

    # Optuna
    p.add_argument("--enable_tuning", action="store_true", default=True)
    p.add_argument("--n_trials",      type=int, default=30)
    p.add_argument("--study_dir",     type=str, default="./result/optuna_study")
    p.add_argument("--study_name",    type=str, default="bert_bin_tuning_mcc")
    p.add_argument("--trials_per_worker", type=int, default=None,
                   help="環境変数TRIALS_PER_WORKERが未設定の場合のみここで指定")
    p.add_argument("--as_master", action="store_true", default=False,
                   help="このプロセスをOptunaのマスターとして最終学習まで実行するか")

    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    # ====== Optuna 無効ならそのまま学習 ======
    if not args.enable_tuning:
        train_and_test(args)
        return

    # ====== Optuna 有効 ======
    os.makedirs(args.study_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(args.study_dir, args.study_name)}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=MedianPruner(n_warmup_steps=3),
    )

    n_trials = args.trials_per_worker or int(os.environ.get("TRIALS_PER_WORKER", args.n_trials))
    study.optimize(lambda t: objective(t, args), n_trials=n_trials, gc_after_trial=True)

    if args.as_master:
        print("\n=== Best Trial ===")
        print(f"number: {study.best_trial.number}")
        print(f"value (eval_mcc): {study.best_trial.value:.5f}")
        print("params:", study.best_trial.params)

        # ベストパラメータ保存
        os.makedirs(args.output_dir, exist_ok=True)
        best_json_path = os.path.join(args.output_dir, "best_params.json")
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"trial_number": study.best_trial.number,
                 "eval_mcc": study.best_trial.value,
                 "params": study.best_trial.params},
                f, indent=2, ensure_ascii=False
            )
        print(f"[INFO] Best parameters saved to: {best_json_path}")

        # 最良試行の成果物を集約コピー（存在すれば）
        best_trial_dir = os.path.join(args.output_dir, f"trial_{study.best_trial.number:03d}")
        best_art_dir   = os.path.join(args.output_dir, "best_trial")
        os.makedirs(best_art_dir, exist_ok=True)
        for fname in ["history.csv", "curve_loss.png", "curve_eval_metrics.png", "val_metrics.json"]:
            src = os.path.join(best_trial_dir, fname)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(best_art_dir, fname))
                    print(f"[INFO] Copied: {src} -> {best_art_dir}")
                except Exception as e:
                    print(f"[WARN] Failed to copy {src}: {e}")

        # ベスト設定で本学習→テスト
        args.best_trial_num = study.best_trial.number
        train_and_test(args, best_params=study.best_trial.params)
    else:
        print("[INFO] Worker finished its share of trials. (No final training on worker)")


if __name__ == "__main__":
    main()
