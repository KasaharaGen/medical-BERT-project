import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed,
    TrainerCallback,  # ★ 追加
)
from transformers.trainer import unwrap_model

import matplotlib.pyplot as plt
import seaborn as sns

# ===== 追加: Optuna =====
import optuna
from optuna.pruners import MedianPruner


# ===== ユーザー環境設定 =====
MODEL_DIR = "../pretraining_bert_2/pretrain_phase2_model_ddp"
TOKENIZER_DIR = "../pretraining_bert_2/pretrain_phase2_tokenizer_ddp"
CSV_PATH = "../data/learning_data.csv"   # 単一CSV（text,label 列）
OUTPUT_DIR = "./result"

SEED = 42
BATCH_SIZE = 16  # 既定値（探索で上書きされ得る）
MAX_LENGTH = 512
LR = 1e-6        # 既定値（探索で上書きされ得る）
NUM_EPOCHS = 5
USE_FP16 = True
VAL_RATIO = 0.1
TEST_RATIO = 0.2
EVAL_STEPS = 50
LOGGING_STEPS = 50

# ==== チューニング設定 ====
ENABLE_TUNING = True        # ← True でOptunaを有効化
N_TRIALS = 20               # 試行数
STUDY_DIR = os.path.join(OUTPUT_DIR, "optuna_study")  # DB保存先
STUDY_NAME = "bert_bin_tuning_mcc"
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
        df, test_size=test_ratio, random_state=seed, stratify=df["label"], shuffle=True
    )
    rel_val_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_rest, test_size=rel_val_ratio, random_state=seed, stratify=train_rest["label"], shuffle=True
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
    def __init__(self, class_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, 2]

        weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def save_history_and_plots(trainer, out_dir: str):
    """学習曲線を同一平面に重ね描きして保存する。rank0のみ出力。"""
    if not trainer.is_world_process_zero():
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # ===== Loss（同一図） =====
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
        plt.xlabel("global_step"); plt.ylabel("loss"); plt.title("Loss (Train & Eval)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_loss.png"), dpi=150); plt.close()

    # ===== Metrics（同一図） =====
    eval_keys = [k for k in ["eval_mcc", "eval_f1", "eval_accuracy"] if k in df.columns and df[k].notna().any()]
    if len(eval_keys) > 0:
        plt.figure()
        for k in eval_keys:
            d = df[df[k].notna()][["step", k]].drop_duplicates(subset="step")
            plt.plot(d["step"], d[k], label=k)
        plt.xlabel("global_step"); plt.ylabel("score"); plt.title("Eval Metrics (MCC / F1 / Accuracy)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_eval_metrics.png"), dpi=150); plt.close()


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
    per_device_batch_size: int,
    use_fp16: bool,
):
    """全rankで集計 → rank0保存（単GPUチューニング時は world_size=1）。"""
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    sampler = DistributedSampler(test_ds, shuffle=False) if is_dist else None
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_fp16 else None)
    loader = DataLoader(test_ds, batch_size=per_device_batch_size, shuffle=False,
                        sampler=sampler, collate_fn=data_collator, num_workers=4, pin_memory=True)

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
                if y == 0 and p == 0: tn += 1
                elif y == 0 and p == 1: fp += 1
                elif y == 1 and p == 0: fn += 1
                else: tp += 1

    cm_local = torch.tensor([tn, fp, fn, tp], device=device, dtype=torch.long)
    if is_dist:
        dist.all_reduce(cm_local, op=dist.ReduceOp.SUM)
    tn, fp, fn, tp = [int(x) for x in cm_local.tolist()]
    total = tn + fp + fn + tp

    if rank == 0:
        acc = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

        test_metrics = {
            "eval_accuracy": acc, "eval_precision": precision, "eval_recall": recall,
            "eval_f1": f1, "eval_mcc": mcc,
            "eval_cm_00": tn, "eval_cm_01": fp, "eval_cm_10": fn, "eval_cm_11": tp,
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
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, square=True,
                    linewidths=0.5, annot_kws={"size": 14, "weight": "bold", "color": "black"})
        plt.title("Test Confusion Matrix (All-Reduce Aggregated)", fontsize=14)
        plt.xlabel("Predicted Label"); plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "test_confusion_matrix.png"), dpi=200)
        plt.close()

        print("[rank0] aggregated test metrics saved.")


# ===== 共通: データ前処理（キャッシュ） =====
_DATASET_CACHE = None
def get_or_build_dataset():
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        dsd = build_datasets_from_single_csv(CSV_PATH, VAL_RATIO, TEST_RATIO, SEED)
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
        def _map(ds):
            ds = ds.map(lambda ex: tokenize_fn(ex, tok), batched=True, remove_columns=["text"])
            ds = ds.rename_column("label", "labels")
            ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
            return ds
        dsd = DatasetDict({
            "train": _map(dsd["train"]),
            "validation": _map(dsd["validation"]),
            "test": _map(dsd["test"]),
        })
        _DATASET_CACHE = (dsd, tok)
    return _DATASET_CACHE


# ===== 目的関数（Optuna） =====
def objective(trial: optuna.Trial) -> float:
    # 単GPU前提：DDPは使わない
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    dsd, tok = get_or_build_dataset()

    # class weight
    labels_np = np.array(dsd["train"]["labels"])
    pos, neg = int(labels_np.sum()), len(labels_np) - int(labels_np.sum())
    class_weight = None
    if pos > 0 and neg > 0:
        w0 = len(labels_np) / (2.0 * neg)
        w1 = len(labels_np) / (2.0 * pos)
        class_weight = torch.tensor([w0, w1], dtype=torch.float)

    # === 探索空間 ===
    lr = trial.suggest_float("learning_rate", 1e-7, 5e-5, log=True)
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    warm = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    dr_hid = trial.suggest_float("hidden_dropout_prob", 0.0, 0.4)
    dr_att = trial.suggest_float("attention_probs_dropout_prob", 0.0, 0.4)
    dr_cls = trial.suggest_float("classifier_dropout", 0.0, 0.5)
    lbl_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)
    sched = trial.suggest_categorical("lr_scheduler_type", ["cosine", "linear", "cosine_with_restarts"])
    per_bs = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    grad_acc = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])

    # model/config
    config = AutoConfig.from_pretrained(
        MODEL_DIR, num_labels=2,
        hidden_dropout_prob=dr_hid,
        attention_probs_dropout_prob=dr_att,
        classifier_dropout=dr_cls,
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8 if USE_FP16 else None)

    # Trial固有の出力先
    trial_out = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
    os.makedirs(trial_out, exist_ok=True)

    args = TrainingArguments(
        output_dir=trial_out,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=per_bs,
        per_device_eval_batch_size=max(8, per_bs),
        learning_rate=lr,
        lr_scheduler_type=sched,
        warmup_ratio=warm,
        weight_decay=wd,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",             # 保存は最後にrank0のみ
        report_to=["none"],
        seed=SEED,
        fp16=USE_FP16,
        dataloader_num_workers=4,
        gradient_accumulation_steps=grad_acc,
        load_best_model_at_end=False,   # 目的関数は最終のevalで十分
        metric_for_best_model="mcc",
        greater_is_better=True,
    )

    # Optuna へ中間値を報告するコールバック（★ TrainerCallback を継承）
    class OptunaReportCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_mcc" in metrics:
                trial.report(metrics["eval_mcc"], step=int(state.global_step))
                if trial.should_prune():
                    raise optuna.TrialPruned()

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weight=class_weight,
        label_smoothing=lbl_smooth,
    )

    # 中間値報告（Optuna連携のためのコールバック）
    trainer.add_callback(OptunaReportCallback())

    trainer.train()
    metrics = trainer.evaluate()
    mcc = float(metrics.get("eval_mcc", 0.0))

    # Trialごとの成果物（最小限）：履歴と曲線
    try:
        save_history_and_plots(trainer, trial_out)
        with open(os.path.join(trial_out, "val_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return mcc


def main_train_once(best_params: dict = None):
    """最良ハイパラで本学習→test集計＆保存"""
    set_all_seeds(SEED)

    # パス検証
    for p, name in [(MODEL_DIR, "MODEL_DIR"), (TOKENIZER_DIR, "TOKENIZER_DIR"), (CSV_PATH, "CSV_PATH")]:
        if name != "CSV_PATH":
            _must_exist_dir(Path(p), name)
        else:
            _must_exist_file(Path(p), name)

    # データ・トークナイザー
    dsd, tok = get_or_build_dataset()

    # class weight
    labels_np = np.array(dsd["train"]["labels"]); pos, neg = int(labels_np.sum()), len(labels_np)-int(labels_np.sum())
    class_weight = None
    if pos > 0 and neg > 0:
        class_weight = torch.tensor([len(labels_np)/(2.0*neg), len(labels_np)/(2.0*pos)], dtype=torch.float)

    # 既定値
    hp = dict(
        learning_rate=LR, weight_decay=0.003, warmup_ratio=0.1,
        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.0, classifier_dropout=0.1,
        label_smoothing=0.1, lr_scheduler_type="cosine",
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=1,
    )
    if best_params:
        hp.update(best_params)

    config = AutoConfig.from_pretrained(
        MODEL_DIR, num_labels=2,
        hidden_dropout_prob=hp["hidden_dropout_prob"],
        attention_probs_dropout_prob=hp["attention_probs_dropout_prob"],
        classifier_dropout=hp["classifier_dropout"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8 if USE_FP16 else None)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=max(8, hp["per_device_train_batch_size"]),
        learning_rate=hp["learning_rate"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
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
        dataloader_num_workers=4,
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        seed=SEED,
        report_to=["none"],
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=dsd["train"], eval_dataset=dsd["validation"],
        tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics,
        class_weight=class_weight, label_smoothing=hp["label_smoothing"],
    )

    # 学習→検証→曲線保存
    trainer.train()
    _ = trainer.evaluate()
    if trainer.is_world_process_zero():
        trainer.save_model(OUTPUT_DIR); tok.save_pretrained(OUTPUT_DIR)
        save_history_and_plots(trainer, OUTPUT_DIR)

    # test（全rank→集約→保存）
    distributed_test_eval_and_save(
        model=trainer.model, test_ds=dsd["test"], tokenizer=tok,
        output_dir=OUTPUT_DIR, per_device_batch_size=hp["per_device_train_batch_size"], use_fp16=USE_FP16,
    )
    print("Done.")


if __name__ == "__main__":
    if ENABLE_TUNING:
        os.makedirs(STUDY_DIR, exist_ok=True)
        storage = f"sqlite:///{os.path.join(STUDY_DIR, STUDY_NAME)}.db"
        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            pruner=MedianPruner(n_warmup_steps=3),
        )

        # ★ 並列ワーカーごとに割り当てる trial 数（合計でN_TRIALSになるよう調整）
        n_trials_this_worker = int(os.environ.get("TRIALS_PER_WORKER", N_TRIALS))

        study.optimize(objective, n_trials=n_trials_this_worker, gc_after_trial=True)

        # ワーカーでは結果のみ更新して終了。マスターだけが本学習を行う。
        is_master = os.environ.get("OPTUNA_MASTER", "0") == "1"

        if is_master:
            print("\n=== Best Trial ===")
            print(f"number: {study.best_trial.number}")
            print(f"value (eval_mcc): {study.best_trial.value:.5f}")
            print("params:", study.best_trial.params)

            # ベストを保存
            best_json_path = os.path.join(OUTPUT_DIR, "best_params.json")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "trial_number": study.best_trial.number,
                    "eval_mcc": study.best_trial.value,
                    "params": study.best_trial.params,
                }, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Best parameters saved to: {best_json_path}")

            # 最良パラメータで最終学習・テスト
            main_train_once(best_params=study.best_trial.params)
        else:
            print("[INFO] Worker finished its share of trials. (No final training on worker)")
    else:
        main_train_once()



