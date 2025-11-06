import os
import random
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch import nn

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)

import matplotlib.pyplot as plt
import seaborn as sns

# ===== ユーザー環境 =====
MODEL_DIR = "../pretraining_bert_2/pretrain_phase2_model_ddp"
TOKENIZER_DIR = "../pretraining_bert_2/pretrain_phase2_tokenizer_ddp"
CSV_PATH = "../data/learning_data.csv"            # 単一CSV（text,label 列）
OUTPUT_DIR = "./result"
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 16                  # 実効は×GPU数（DDP）
LR = 2e-5
NUM_EPOCHS = 3
USE_FP16 = True                  # Quadro想定でfp16
VAL_RATIO = 0.1                  # validation 割合（残りから層化分割）
TEST_RATIO = 0.1                 # test 割合（最初に層化分割）
EVAL_STEPS = 200
LOGGING_STEPS = 50
# ======================

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); set_seed(seed)

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
    assert {"text","label"} <= set(df.columns), "CSVに text,label 列が必要である。"
    train_df, val_df, test_df = stratified_three_split(df[["text","label"]], test_ratio, val_ratio, seed)
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
    return {"accuracy": acc, "f1": f1, "mcc": mcc,
            "cm_00": cm[0][0], "cm_01": cm[0][1], "cm_10": cm[1][0], "cm_11": cm[1][1]}

class WeightedTrainer(Trainer):
    def __init__(self, class_weight: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs); self.class_weight = class_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        transformers>=4.41 などが渡す num_items_in_batch 等の追加kwに対応するため **kwargs を受ける。
        """
        labels = inputs.get("labels")
        # labelsをモデルに渡すと内部で標準lossが計算されるため、重み付けしたい場合は除外して自前計算
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, 2]

        weight = None
        if self.class_weight is not None:
            # cuda でも cpu でも動くようにデバイス合わせ
            weight = self.class_weight.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
def save_history_and_plots(trainer, out_dir: str):
    """train と valid のみ曲線保存。test は図を作らない。"""
    if not trainer.is_world_process_zero():
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # train loss
    if "loss" in df.columns:
        d = df[df["loss"].notna()][["step","loss"]].drop_duplicates(subset="step")
        if len(d) > 0:
            plt.figure()
            plt.plot(d["step"], d["loss"])
            plt.xlabel("global_step"); plt.ylabel("train_loss"); plt.title("Train Loss")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_train_loss.png"), dpi=150); plt.close()

    # valid 指標のみ
    for key in ["eval_mcc","eval_f1","eval_accuracy","eval_loss"]:
        if key in df.columns:
            d = df[df[key].notna()][["step", key]].drop_duplicates(subset="step")
            if len(d) > 0:
                plt.figure()
                plt.plot(d["step"], d[key])
                plt.xlabel("global_step"); plt.ylabel(key); plt.title(key)
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"curve_{key}.png"), dpi=150); plt.close()

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

def main():
    set_all_seeds(SEED)

    # ---- パスの型と存在チェック ----
    model_dir = _must_be_str_path(MODEL_DIR, "MODEL_DIR")
    tok_dir   = _must_be_str_path(TOKENIZER_DIR, "TOKENIZER_DIR")
    out_dir   = _must_be_str_path(OUTPUT_DIR, "OUTPUT_DIR")
    csv_path  = _must_be_str_path(CSV_PATH, "CSV_PATH")

    _must_exist_dir(model_dir, "MODEL_DIR")
    _must_exist_dir(tok_dir, "TOKENIZER_DIR")
    _must_exist_file(csv_path, "CSV_PATH（単一CSV）")

    os.makedirs(out_dir, exist_ok=True)
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
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        return ds
    dsd["train"] = _map(dsd["train"])
    dsd["validation"] = _map(dsd["validation"])
    dsd["test"] = _map(dsd["test"])

    # 4) class weight は train のみから計算
    labels_np = np.array(dsd["train"]["labels"])
    pos = int(labels_np.sum()); neg = len(labels_np) - pos
    class_weight = None
    if pos > 0 and neg > 0:
        w0 = len(labels_np)/(2.0*neg); w1 = len(labels_np)/(2.0*pos)
        class_weight = torch.tensor([w0, w1], dtype=torch.float)

    # 5) モデル
    config = AutoConfig.from_pretrained(MODEL_DIR, num_labels=2)
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
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
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

    # 10) test で最終評価（図は保存しない）
    test_metrics = trainer.evaluate(dsd["test"])

    # 11) 成果物保存（rank0のみ）
    if trainer.is_world_process_zero():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # モデルとトークナイザー
        trainer.save_model(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)

        # 履歴と曲線（train/validのみ）
        save_history_and_plots(trainer, OUTPUT_DIR)

        # test結果をJSON/CSVで保存（図は作らない）
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        print(f"Artifacts saved under: {OUTPUT_DIR}")
        print("history.csv / curve_train_loss.png / curve_eval_*.png / test_metrics.json などを確認せよ。")

            # 混同行列を別CSVに保存（test）
        if all(k in test_metrics for k in ["eval_cm_00","eval_cm_01","eval_cm_10","eval_cm_11"]):
            cm = np.array([
                [test_metrics["eval_cm_00"], test_metrics["eval_cm_01"]],
                [test_metrics["eval_cm_10"], test_metrics["eval_cm_11"]],
            ], dtype=int)

            # CSV 保存（表形式）
            cm_df = pd.DataFrame(
                cm,
                index=["True_0","True_1"],
                columns=["Pred_0","Pred_1"]
            )
            cm_csv_path = os.path.join(OUTPUT_DIR, "test_confusion_matrix.csv")
            cm_df.to_csv(cm_csv_path, index=True)

            # 追加: seaborn ヒートマップとして PNG 保存
            plt.figure(figsize=(5,4))
            ax = sns.heatmap(
                cm_df,
                annot=True, fmt="d",
                cmap="Blues",
                cbar=True,
                square=True,
                linewidths=0.5,
                annot_kws={"size":14, "weight":"bold", "color":"black"},
            )
            ax.set_title("Test Confusion Matrix (Seaborn)", fontsize=14)
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            plt.tight_layout()
            cm_png_path = os.path.join(OUTPUT_DIR, "test_confusion_matrix.png")
            plt.savefig(cm_png_path, dpi=200)
            plt.close()

            print(f"\nConfusion matrix saved:\n- {cm_csv_path}\n- {cm_png_path}")


if __name__ == "__main__":
    main()
