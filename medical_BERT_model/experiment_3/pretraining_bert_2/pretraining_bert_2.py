import os
import random
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import (
    PreTrainedTokenizerFast,
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from torch.optim import AdamW
import optuna
from optuna.pruners import MedianPruner
from multiprocessing import Manager
import matplotlib.pyplot as plt


# ===============================
#  ユーティリティ
# ===============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===============================
#  Dataset
# ===============================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        return {"input_ids": enc["input_ids"]}


# ===============================
#  DDP Worker （各GPUプロセス）
# ===============================
def ddp_worker(rank: int, world_size: int, hparams: dict, return_dict, save_model: bool):
    """
    rank:       プロセス / GPU 番号
    world_size: 全GPU数
    hparams:    ハイパーパラメータ一式（dict） + （探索時は）trial
    return_dict: rank0 から best_eval_loss / pruned フラグを返す共有辞書
    save_model: True のときだけモデル・lossカーブを保存する（best trial 再学習時のみ）
    """
    try:
        # === 通信初期化 ===
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        set_seed(42 + rank)

        # === データ読み込み ===
        df = pd.read_csv("../data/dengue_data/dengue_sentences.csv")
        texts = df["sentence"].dropna().astype(str).tolist()
        split_idx = int(len(texts) * 0.98)
        train_texts, eval_texts = texts[:split_idx], texts[split_idx:]

        # === Tokenizer ===
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "../pretraining_bert_1/pretraining_bert_best/best_model"
        )

        # === Dataset / DataLoader ===
        max_length = 512
        mlm_probability = 0.15
        batch_size = hparams["batch_size"]

        train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
        eval_dataset = TextDataset(eval_texts, tokenizer, max_length=max_length)

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=True, mlm_probability=mlm_probability
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=eval_sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

        # === モデルロード（Dropout等は hparams で上書き） ===
        config = BertConfig.from_pretrained(
            "../pretraining_bert_1/pretraining_bert_best/best_model"
        )
        config.hidden_dropout_prob = hparams["hidden_dropout"]
        config.attention_probs_dropout_prob = hparams["attention_dropout"]

        model = BertForMaskedLM.from_pretrained(
            "../pretraining_bert_1/pretraining_bert_best/best_model",
            config=config,
        ).to(device)
        model = DDP(model, device_ids=[rank])

        # === Optimizer & Scheduler ===
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=hparams["learning_rate"],
        )

        num_epochs = hparams["num_epochs"]
        patience = hparams["patience"]
        min_delta = hparams["min_delta"]

        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps = int(max_train_steps * hparams["warmup_ratio"])

        lr_scheduler = get_scheduler(
            name=hparams["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        best_eval_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False
        pruned = False

        train_loss_history = []
        eval_loss_history = []

        # === 学習ループ ===
        for epoch in range(num_epochs):
            # ---- Train ----
            model.train()
            train_sampler.set_epoch(epoch)

            total_train_loss = 0.0
            num_train_steps = 0

            for batch in tqdm(
                train_loader,
                desc=f"[Rank {rank}] Epoch {epoch+1} Train",
                leave=False,
            ):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_train_loss += loss.item()
                num_train_steps += 1

            local_avg_train_loss = total_train_loss / max(1, num_train_steps)

            # 全GPUで平均 train loss を計算
            train_loss_tensor = torch.tensor(local_avg_train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (train_loss_tensor / world_size).item()

            # ---- Eval (全rankで実行) ----
            model.eval()
            total_eval_loss = 0.0
            num_eval_steps = 0

            with torch.no_grad():
                for batch in tqdm(
                    eval_loader,
                    desc=f"[Rank {rank}] Epoch {epoch+1} Eval",
                    leave=False,
                ):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_eval_loss += loss.item()
                    num_eval_steps += 1

            loss_tensor = torch.tensor(total_eval_loss, device=device)
            steps_tensor = torch.tensor(float(num_eval_steps), device=device)

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)

            avg_eval_loss = (loss_tensor / steps_tensor).item()

            # ---- rank0 で early stopping & pruner 判定 & ログ ----
            prune_flag = False
            if rank == 0:
                print(
                    f"[Phase2] Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, "
                    f"Eval Loss: {avg_eval_loss:.6f}"
                )

                if save_model:
                    train_loss_history.append(avg_train_loss)
                    eval_loss_history.append(avg_eval_loss)

                # Early Stopping（patience, min_delta）
                if avg_eval_loss < best_eval_loss - min_delta:
                    best_eval_loss = avg_eval_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[Phase2] Early stopping triggered at epoch {epoch+1}")
                        early_stop = True

                # Optuna pruner（探索フェーズのみ）
                if ("trial" in hparams) and (not save_model):
                    trial = hparams["trial"]
                    trial.report(avg_eval_loss, step=epoch)
                    if trial.should_prune():
                        print(f"[Phase2] Trial pruned at epoch {epoch+1}")
                        prune_flag = True

            # early_stop / prune を全rankにブロードキャスト
            early_stop_tensor = torch.tensor(1 if early_stop else 0, device=device)
            prune_tensor = torch.tensor(1 if prune_flag else 0, device=device)

            dist.broadcast(early_stop_tensor, src=0)
            dist.broadcast(prune_tensor, src=0)

            early_stop = bool(early_stop_tensor.item())
            prune_flag = bool(prune_tensor.item())

            # prune 優先
            if prune_flag:
                pruned = True
                if rank == 0:
                    return_dict["best_eval_loss"] = float(avg_eval_loss)
                    return_dict["pruned"] = True
                dist.destroy_process_group()
                return

            if early_stop:
                break

        # rank0 から結果を親に返す
        if rank == 0 and not pruned:
            return_dict["best_eval_loss"] = float(best_eval_loss)
            return_dict["pruned"] = False

            # save_model=True のときのみモデル・ログ保存（Phase1と同構造）
            if save_model:
                save_dir = hparams["save_dir"]
                os.makedirs(save_dir, exist_ok=True)

                model_dir = os.path.join(save_dir, "best_model")
                tok_dir   = os.path.join(save_dir, "tokenizer")
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(tok_dir, exist_ok=True)

                # モデル本体
                model.module.save_pretrained(model_dir)
                # トークナイザ
                tokenizer.save_pretrained(tok_dir)

                # loss ログ（loss_log.csv）
                pd.DataFrame(
                    {
                        "epoch": list(range(1, len(train_loss_history) + 1)),
                        "train_loss": train_loss_history,
                        "eval_loss": eval_loss_history,
                    }
                ).to_csv(
                    os.path.join(save_dir, "loss_log.csv"),
                    index=False,
                )

                # loss curve（loss_curve.png）
                plt.figure(figsize=(8, 5))
                plt.plot(
                    range(1, len(train_loss_history) + 1),
                    train_loss_history,
                    label="Train Loss",
                )
                plt.plot(
                    range(1, len(eval_loss_history) + 1),
                    eval_loss_history,
                    label="Eval Loss",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Phase2 DDP Training Loss")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "loss_curve.png"))
                plt.close()

        dist.destroy_process_group()

    except Exception as e:
        print(f"[Rank {rank}] エラー発生: {e}")
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        if rank == 0:
            return_dict["best_eval_loss"] = float("inf")
            return_dict["pruned"] = False


# ===============================
#  Optuna 用ハイパーパラ構築（Phase2用）
# ===============================
def build_hparams_from_trial(trial: optuna.trial.Trial) -> dict:
    hparams = {}

    # 学習率・正則化
    hparams["learning_rate"] = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    hparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-5, log=True)

    # warmup割合
    hparams["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.05)

    # MLM マスク割合
    #hparams["mlm_probability"] = trial.suggest_float("mlm_probability", 0.15, 0.25)

    # バッチサイズ
    hparams["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])

    # max_length
    #hparams["max_length"] = trial.suggest_categorical("max_length", [256, 512])

    # dropout 系
    hparams["hidden_dropout"] = trial.suggest_float("hidden_dropout", 0.1, 0.4)
    hparams["attention_dropout"] = trial.suggest_float("attention_dropout", 0.1, 0.3)

    # スケジューラ
    hparams["lr_scheduler_type"] = trial.suggest_categorical("lr_scheduler_type",["linear", "cosine"],)

    # Epoch / Early stopping
    hparams["num_epochs"] = trial.suggest_int("num_epochs", 5, 8)
    hparams["patience"] = trial.suggest_int("patience", 2, 3)
    hparams["min_delta"] = trial.suggest_float("min_delta", 1e-5, 5e-4, log=True)

    return hparams


# ===============================
#  Optuna Objective
# ===============================
def objective(trial: optuna.trial.Trial) -> float:
    hparams = build_hparams_from_trial(trial)

    # save_dir は探索中は使わないがキーだけ入れておく
    hparams["save_dir"] = "dummy_dir"
    # pruner用に trial を埋め込む
    hparams["trial"] = trial

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("CUDA GPU が見つからない")

    with Manager() as manager:
        return_dict = manager.dict()

        mp.spawn(
            ddp_worker,
            args=(world_size, hparams, return_dict, False),
            nprocs=world_size,
            join=True,
        )

        best_eval_loss = float(return_dict.get("best_eval_loss", float("inf")))
        pruned = bool(return_dict.get("pruned", False))

    if pruned:
        raise optuna.TrialPruned()

    return best_eval_loss


# ===============================
#  エントリーポイント
# ===============================
if __name__ == "__main__":
    # DDP 用環境変数（ローカル）
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_DEBUG"] = "WARN"

    set_seed(42)

    # ===== Optuna 探索（Phase2 ドメイン学習） =====
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_warmup_steps=1),
    )

    # 試行回数は環境に応じて調整
    study.optimize(objective, n_trials=10)

    print("========== Phase2 Optuna Result ==========")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best eval loss   : {study.best_trial.value}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # ===== ベスト trial のパラメータで再学習し、ここで初めて保存 =====
    best_trial = study.best_trial

    class DummyTrial:
        """best_trial.params から hparams を再構成するための簡易ラッパ"""
        def __init__(self, params):
            self.params = params

        def suggest_float(self, name, low, high, log=False):
            return float(self.params[name])

        def suggest_categorical(self, name, choices):
            return self.params[name]

        def suggest_int(self, name, low, high):
            return int(self.params[name])

    dummy_trial = DummyTrial(best_trial.params)
    best_hparams = build_hparams_from_trial(dummy_trial)

    # Phase2 出力先（Phase1 と同じ構造：best_model, tokenizer, loss_log, loss_curve）
    best_hparams["save_dir"] = "pretraining_bert_best"

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("CUDA GPU が見つからない")

    print("===== Phase2: Training final model with best hyperparameters and saving it =====")

    with Manager() as manager:
        return_dict = manager.dict()
        mp.spawn(
            ddp_worker,
            args=(world_size, best_hparams, return_dict, True),
            nprocs=world_size,
            join=True,
        )

    print("Phase2 final model saved in: pretraining_bert_best")
