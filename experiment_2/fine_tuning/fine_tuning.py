import os, math, json, random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Any

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, precision_recall_curve,
    confusion_matrix, matthews_corrcoef
)

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# ========= ユーザ設定 =========
MODEL_DIR = "../pretraining_bert_2/pretrain_phase2_model_ddp"      # 事前学習済みBERTのディレクトリ
TOKENIZER_DIR = "../pretraining_bert_2/pretrain_phase2_tokenizer_ddp"
DATA_CSV  = "../ehr_dengue_binary.csv"  # text,label,(optional)patient_id
OUT_DIR   = "./results"          # 出力先
NUM_EPOCHS = 6
LR = 2e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
BATCH_TRAIN = 16
BATCH_EVAL  = 64
EARLY_STOP_PATIENCE = 2
SEED = 2025
USE_BF16 = False  # Ampere以降ならTrue推奨
# =============================

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ---- データ読み込み ----
df = pd.read_csv(DATA_CSV)
assert "text" in df.columns and "label" in df.columns, "CSVに text, label 列が必要である"
df = df.dropna(subset=["text", "label"]).copy()
df["label"] = df["label"].astype(int)
assert set(df["label"].unique()) <= {0,1}, "labelは0/1である必要がある"

# ---- 分割（患者IDあればGroup分割） ----
def group_split(df, test_size=0.1, valid_size=0.1, seed=SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train_tmp, idx_test = next(gss1.split(df, groups=df["patient_id"]))
    df_train_tmp = df.iloc[idx_train_tmp].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_size/(1.0 - test_size), random_state=seed)
    idx_train, idx_valid = next(gss2.split(df_train_tmp, groups=df_train_tmp["patient_id"]))
    df_train = df_train_tmp.iloc[idx_train].reset_index(drop=True)
    df_valid = df_train_tmp.iloc[idx_valid].reset_index(drop=True)
    return df_train, df_valid, df_test

def stratified_split(df, test_size=0.1, valid_size=0.1, seed=SEED):
    from sklearn.model_selection import StratifiedShuffleSplit
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train_tmp, idx_test = next(sss1.split(df, df["label"]))
    df_train_tmp = df.iloc[idx_train_tmp].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=valid_size/(1.0 - test_size), random_state=seed)
    idx_train, idx_valid = next(sss2.split(df_train_tmp, df_train_tmp["label"]))
    df_train = df_train_tmp.iloc[idx_train].reset_index(drop=True)
    df_valid = df_train_tmp.iloc[idx_valid].reset_index(drop=True)
    return df_train, df_valid, df_test

has_pid = "patient_id" in df.columns
if has_pid:
    df_train, df_valid, df_test = group_split(df)
else:
    df_train, df_valid, df_test = stratified_split(df)

print(f"[Split] train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}, has_pid={has_pid}")

# ---- HF Datasets へ変換 ----
raw = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "valid": Dataset.from_pandas(df_valid),
    "test":  Dataset.from_pandas(df_test),
})

# ---- トークナイザ／モデル ----
tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

# ---- 前処理 ----
def tok_fn(examples):
    out = tok(examples["text"], truncation=True, max_length=512)
    out["labels"] = examples["label"]
    return out

cols_keep = [c for c in raw["train"].column_names if c in ("text","label","patient_id")]
tokenized = raw.map(tok_fn, batched=True, remove_columns=[c for c in raw["train"].column_names if c not in cols_keep])

# ---- クラス重み ----
y_train = np.array(tokenized["train"]["labels"])
pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
pos_w = (neg / max(1, pos)) if pos > 0 else 1.0
class_weights = torch.tensor([1.0, float(pos_w)])

# ---- 重み付きloss差し替え ----
_old_forward = model.forward
def forward_with_weights(**kwargs):
    labels = kwargs.get("labels", None)
    outputs = _old_forward(**kwargs)
    if labels is not None:
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return type(outputs)(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    return outputs
model.forward = forward_with_weights

# ---- 評価指標（MCCを主指標） ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)  # 評価時は0.5固定（最終は別途閾値最適化可）
    # MCC
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    # F1
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    # 混同行列（要素で返す）
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    # 参考で確率系も保存（監視用）
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    auprc = average_precision_score(labels, probs)
    return {
        "mcc": float(mcc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "auroc": float(auroc), "auprc": float(auprc)
    }

# ---- Trainer ----
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    lr_scheduler_type="linear",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mcc",     # ★ MCCを主指標
    greater_is_better=True,
    fp16=(not USE_BF16),
    bf16=USE_BF16,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    group_by_length=True,
    save_total_limit=3,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["valid"],
    tokenizer=tok,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)]
)

# ---- 学習 ----
train_out = trainer.train()

# ---- ベストモデルを保存（safetensorsで） ----
best_dir = os.path.join(OUT_DIR, "best")
os.makedirs(best_dir, exist_ok=True)
trainer.save_model(best_dir)  # save_pretrained を内部で呼ぶ
tok.save_pretrained(best_dir)
# 念のため safe_serialization 明示（transformers>=4.44）
trainer.model.save_pretrained(best_dir, safe_serialization=True)

# ---- 学習曲線を保存 ----
# log_history から loss と eval_mcc 等を抽出
hist = trainer.state.log_history
epochs = []
train_losses = []
eval_epochs = []
eval_mcc = []
eval_f1 = []
for h in hist:
    if "loss" in h and "epoch" in h:
        epochs.append(h["epoch"]); train_losses.append(h["loss"])
    if "eval_mcc" in h and "epoch" in h:
        eval_epochs.append(h["epoch"]); eval_mcc.append(h["eval_mcc"])
    if "eval_f1" in h and "epoch" in h:
        eval_f1.append(h["eval_f1"])

plt.figure(figsize=(8,5))
if len(epochs) > 0:
    plt.plot(epochs, train_losses, label="train_loss")
if len(eval_epochs) > 0:
    plt.plot(eval_epochs, eval_mcc, label="eval_mcc")
if len(eval_epochs) > 0 and len(eval_f1) == len(eval_epochs):
    plt.plot(eval_epochs, eval_f1, label="eval_f1")
plt.xlabel("epoch"); plt.ylabel("value"); plt.title("Learning Curves (loss/MCC/F1)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=150)
plt.close()

# ---- Valid/Test で詳細評価・混同行列図を保存 ----
def eval_and_save(name: str, dataset, out_dir: str) -> Dict[str, Any]:
    pred = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()[:,1]
    labels = pred.label_ids
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    auprc = average_precision_score(labels, probs)

    # 可視化
    import seaborn as sns
    cm = np.array([[tn, fp],[fn, tp]])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=["pred 0","pred 1"], yticklabels=["true 0","true 1"])
    plt.title(f"Confusion Matrix ({name})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close()

    return {
        f"{name}_tn": int(tn), f"{name}_fp": int(fp), f"{name}_fn": int(fn), f"{name}_tp": int(tp),
        f"{name}_mcc": float(mcc), f"{name}_f1": float(f1),
        f"{name}_precision": float(prec), f"{name}_recall": float(rec),
        f"{name}_auroc": float(auroc), f"{name}_auprc": float(auprc)
    }

valid_report = eval_and_save("valid", tokenized["valid"], OUT_DIR)
test_report  = eval_and_save("test",  tokenized["test"],  OUT_DIR)

final_report = {
    "train_size": len(df_train),
    "valid_size": len(df_valid),
    "test_size": len(df_test),
    "has_patient_id": bool(has_pid),
    "class_weights": [1.0, float(pos_w)],
    "best_checkpoint": best_dir,
}
final_report.update(valid_report); final_report.update(test_report)

with open(os.path.join(OUT_DIR, "threshold_and_metrics.json"), "w") as f:
    json.dump(final_report, f, indent=2)

print("[Final]", json.dumps(final_report, ensure_ascii=False, indent=2))
