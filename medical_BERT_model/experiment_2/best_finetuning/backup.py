# -*- coding: utf-8 -*-
# ファイル名例: fine_tuning_distributed_test.py

import os
import random
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)
from transformers.trainer import unwrap_model

import matplotlib.pyplot as plt
import seaborn as sns


# ===== ユーザー環境設定 =====
MODEL_DIR = "../pretraining_bert_2/pretrain_phase2_model_ddp"
TOKENIZER_DIR = "../pretraining_bert_2/pretrain_phase2_tokenizer_ddp"
CSV_PATH = "../data/learning_data.csv"   # 単一CSV（text,label 列）
OUTPUT_DIR = "./result"

SEED = 42
BATCH_SIZE = 16 #best
MAX_LENGTH = 512
LR = 1e-6
NUM_EPOCHS = 3
USE_FP16 = True                # Quadro想定でfp16
VAL_RATIO = 0.1                # validation 割合（残りから層化分割）
TEST_RATIO = 0.2               # test 割合（最初に層化分割）
EVAL_STEPS = 10
LOGGING_STEPS = 50
# ===========================


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def stratified_three_split(
    df: pd.DataFrame, test_ratio: float, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """まず test を層化で切り出し、その後の残りから val を層化で切り出す。"""
    assert 0 < test_ratio < 0.5 and 0 < val_ratio < 0.5 and test_ratio + val_ratio < 1.0
    train_rest, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["label"]
    )
    # 残りから val を切る（残りに対する相対割合）
    rel_val_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_rest, test_size=rel_val_ratio, random_state=seed, stratify=train_rest["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_datasets_from_single_csv(csv_path: str, val_ratio: float, test_ratio: float, seed: int) -> DatasetDict:
    df = pd.read_csv(csv_path)
    assert {"text", "label"} <= set(df.columns), "CSVに text,label 列が必要である。"
    train_df, val_df, test_df = stratified_three_split(df[["text", "label"]], test_ratio, val_ratio, seed)
    ds_tr = Dataset.from_pandas(train_df)
    ds_va = Dataset.from_pandas(val_df)
    ds_te = Dataset.from_pandas(test_df)
    return DatasetDict({"train": ds_tr, "validation": ds_va, "test": ds_te})


def tokenize_fn(ex, tok):
    return tok(ex["text"], truncation=True, max_length=MAX_LENGTH, padding=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()
    return {
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "cm_00": cm[0][0],
        "cm_01": cm[0][1],
        "cm_10": cm[1][0],
        "cm_11": cm[1][1],
    }


class WeightedTrainer(Trainer):
    """クラス不均衡対策でCrossEntropyにclass_weightを適用するTrainer拡張"""
    def __init__(self, class_weight: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        transformers>=4.41 などが渡す追加kwargs（num_items_in_batch 等）に対応。
        """
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, 2]

        weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
        #loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def save_history_and_plots(trainer, out_dir: str):
    """学習曲線を同一平面に重ね描きして保存する。rank0のみ出力。"""
    if not trainer.is_world_process_zero():
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # ===== Loss: train と eval を同一図に =====
    # train loss は "loss"、eval loss は "eval_loss"
    df_train_loss = df[df.get("loss").notna()] if "loss" in df.columns else pd.DataFrame()
    df_eval_loss  = df[df.get("eval_loss").notna()] if "eval_loss" in df.columns else pd.DataFrame()

    if len(df_train_loss) > 0 or len(df_eval_loss) > 0:
        plt.figure()
        if len(df_train_loss) > 0:
            d = df_train_loss[["step", "loss"]].drop_duplicates(subset="step")
            plt.plot(d["step"], d["loss"], label="train_loss")
        if len(df_eval_loss) > 0:
            d = df_eval_loss[["step", "eval_loss"]].drop_duplicates(subset="step")
            plt.plot(d["step"], d["eval_loss"], label="eval_loss")
        plt.xlabel("global_step")
        plt.ylabel("loss")
        plt.title("Loss (Train & Eval)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_loss.png"), dpi=150)
        plt.close()

    # ===== Eval Metrics: mcc / f1 / accuracy を同一図に =====
    eval_keys = []
    for k in ["eval_mcc", "eval_f1", "eval_accuracy"]:
        if k in df.columns and df[k].notna().any():
            eval_keys.append(k)

    if len(eval_keys) > 0:
        plt.figure()
        for k in eval_keys:
            d = df[df[k].notna()][["step", k]].drop_duplicates(subset="step")
            plt.plot(d["step"], d[k], label=k)
        plt.xlabel("global_step")
        plt.ylabel("score")
        plt.title("Eval Metrics (MCC / F1 / Accuracy)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_eval_metrics.png"), dpi=150)
        plt.close()


def _must_be_str_path(p, name: str) -> Path:
    if p is None:
        raise RuntimeError(f"{name} が None だ。文字列パスを指定すべきである。")
    if not isinstance(p, (str, bytes, os.PathLike)):
        raise RuntimeError(f"{name} は str/Path であるべきだが、{type(p)} が渡された。")
    return Path(p)


def _must_exist_dir(p: Path, name: str) -> Path:
    if not p.exists():
        raise RuntimeError(f"{name} が存在しない: {p}")
    if not p.is_dir():
        raise RuntimeError(f"{name} はディレクトリであるべきだが、ファイルだった: {p}")
    return p


def _must_exist_file(p: Path, name: str) -> Path:
    if not p.exists():
        raise RuntimeError(f"{name} が存在しない: {p}")
    if not p.is_file():
        raise RuntimeError(f"{name} はファイルであるべきだが、ディレクトリだった: {p}")
    return p


@torch.no_grad()
def distributed_test_eval_and_save(
    model,
    test_ds,
    tokenizer,
    output_dir: str,
    per_device_batch_size: int = BATCH_SIZE,
    use_fp16: bool = True,
):
    """
    各 rank が自分の分担テストデータで推論し、TN/FP/FN/TP を数える。
    → all_reduce(SUM)で合算 → rank0 が指標算出と保存（JSON/CSV/PNG）を行う。
    """
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    sampler = DistributedSampler(test_ds, shuffle=False) if is_dist else None
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_fp16 else None)
    loader = DataLoader(
        test_ds,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    device = f"cuda:{rank}" if torch.cuda.is_available() and is_dist else ("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = unwrap_model(model).to(device)
    base_model.eval()

    tn = fp = fn = tp = 0

    amp_ctx = torch.cuda.amp.autocast(enabled=use_fp16 and torch.cuda.is_available())
    with amp_ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()

            logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(-1).cpu().numpy()

            for y, p in zip(labels, preds):
                if y == 0 and p == 0:
                    tn += 1
                elif y == 0 and p == 1:
                    fp += 1
                elif y == 1 and p == 0:
                    fn += 1
                else:
                    tp += 1

    cm_local = torch.tensor([tn, fp, fn, tp], device=device, dtype=torch.long)
    if is_dist:
        dist.all_reduce(cm_local, op=dist.ReduceOp.SUM)
    tn, fp, fn, tp = [int(x) for x in cm_local.tolist()]
    total = tn + fp + fn + tp

    if rank == 0:
        acc = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

        test_metrics = {
            "eval_accuracy": acc,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
            "eval_mcc": mcc,
            "eval_cm_00": tn,
            "eval_cm_01": fp,
            "eval_cm_10": fn,
            "eval_cm_11": tp,
            "world_size": world_size,
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        pd.DataFrame([test_metrics]).to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

        cm = np.array([[tn, fp], [fn, tp]], dtype=int)
        cm_df = pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
        cm_df.to_csv(os.path.join(output_dir, "test_confusion_matrix.csv"))

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 14, "weight": "bold", "color": "black"},
        )
        plt.title("Test Confusion Matrix (All-Reduce Aggregated)", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "test_confusion_matrix.png"), dpi=200)
        plt.close()

        print("[rank0] aggregated test metrics saved.")


def main():
    set_all_seeds(SEED)

    # ---- パスの型と存在チェック ----
    model_dir = _must_be_str_path(MODEL_DIR, "MODEL_DIR")
    tok_dir = _must_be_str_path(TOKENIZER_DIR, "TOKENIZER_DIR")
    out_dir = _must_be_str_path(OUTPUT_DIR, "OUTPUT_DIR")
    csv_path = _must_be_str_path(CSV_PATH, "CSV_PATH")

    _must_exist_dir(model_dir, "MODEL_DIR")
    _must_exist_dir(tok_dir, "TOKENIZER_DIR")
    _must_exist_file(csv_path, "CSV_PATH（単一CSV）")

    os.makedirs(out_dir, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[INFO] model_dir={model_dir.resolve()}")
        print(f"[INFO] tokenizer_dir={tok_dir.resolve()}")
        print(f"[INFO] output_dir={out_dir.resolve()}")
        print(f"[INFO] csv_path={csv_path.resolve()}")

    # 1) データ（三分割）
    dsd = build_datasets_from_single_csv(CSV_PATH, VAL_RATIO, TEST_RATIO, SEED)

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)

    # 3) tokenize → torch format（train/valid/test 全て）
    def _map(ds):
        ds = ds.map(lambda ex: tokenize_fn(ex, tok), batched=True, remove_columns=["text"])
        ds = ds.rename_column("label", "labels")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    dsd["train"] = _map(dsd["train"])
    dsd["validation"] = _map(dsd["validation"])
    dsd["test"] = _map(dsd["test"])

    # 4) class weight は train のみから計算（inverse-frequency）
    labels_np = np.array(dsd["train"]["labels"])
    pos = int(labels_np.sum())
    neg = len(labels_np) - pos
    class_weight = None
    if pos > 0 and neg > 0:
        w0 = len(labels_np) / (2.0 * neg)
        w1 = len(labels_np) / (2.0 * pos)
        class_weight = torch.tensor([w0, w1], dtype=torch.float)

    # 5) モデル
    config = AutoConfig.from_pretrained(MODEL_DIR, num_labels=2,hidden_dropout_prob=0.2,attention_probs_dropout_prob=0,classifier_dropout=0.1)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

    # 6) Collator
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8 if USE_FP16 else None)

    # 7) 学習設定（DDPはtorchrunで自動）
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        #warmup_ratio=0.08,(best)
        warmup_ratio=0.1,
        #weight_decay=0.003,(best)
        weight_decay=0.003,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",      
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        #metric_for_best_model="mcc",
        metric_for_best_model="mcc",
        greater_is_better=True,
        fp16=USE_FP16,
        bf16=False,
        gradient_accumulation_steps=1,
        report_to=["none"],
        seed=SEED,
        dataloader_num_workers=4,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weight=class_weight,
    )

    # 8) 学習
    trainer.train()

    # 9) valid で最終評価（ベスト重み）
    _ = trainer.evaluate()  # dsd["validation"]

    # 10) test は全rankで実行 → all_reduce 集約 → rank0が保存
    distributed_test_eval_and_save(
        model=trainer.model,
        test_ds=dsd["test"],
        tokenizer=tok,
        output_dir=OUTPUT_DIR,
        per_device_batch_size=BATCH_SIZE,
        use_fp16=USE_FP16,
    )

    # 11) モデルと履歴・曲線の保存（rank0のみ）
    if trainer.is_world_process_zero():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        save_history_and_plots(trainer, OUTPUT_DIR)
        print(f"Artifacts saved under: {OUTPUT_DIR}")
        print("history.csv / curve_train_loss.png / curve_eval_*.png / test_metrics.(json,csv) / test_confusion_matrix.(csv,png)")


if __name__ == "__main__":
    main()
