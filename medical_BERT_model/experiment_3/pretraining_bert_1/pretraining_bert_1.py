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
from torch.utils.data import Subset

from transformers import (
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

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
        # DDP 初期化
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

        # ==========================
        #  データ読み込み
        # ==========================
        df = pd.read_csv("../data/infection_data/infection_sentences.csv")
        sentences = df["sentence"].dropna().astype(str).tolist()

        train_ratio = 0.98
        train_size = int(len(sentences) * train_ratio)
        train_texts = sentences[:train_size]
        eval_texts = sentences[train_size:]

        # ==========================
        #  Tokenizer
        # ==========================
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="../tokenizer/tokenizer.json",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        # ==========================
        #  Dataset / DataLoader
        # ==========================
        max_length = 512
        mlm_probability = 0.15
        batch_size = 64  # 必要ならここを Optuna 対象に戻してよい

        train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
        eval_dataset  = TextDataset(eval_texts,  tokenizer, max_length=max_length)

        # デバッグ／スモークテスト用のサブセット（不要なら外す）
        #train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
        #eval_dataset  = Subset(eval_dataset,  range(min(1000, len(eval_dataset))))

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
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

        # ==========================
        #  モデル構築（BERT-base 固定）
        # ==========================
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,                 # 固定
            num_hidden_layers=12,            # 固定
            num_attention_heads=12,          # 固定（768 / 12 = 64）
            intermediate_size=3072,          # 固定（標準は 4 * hidden）
            max_position_embeddings=512,
            hidden_dropout_prob=hparams["hidden_dropout"],
            attention_probs_dropout_prob=hparams["attention_dropout"],
        )

        model = BertForMaskedLM(config).to(device)
        model = DDP(model, device_ids=[rank])

        # ==========================
        #  Optimizer / Scheduler
        # ==========================
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        num_epochs = 5
        patience   = hparams["patience"]

        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps    = int(max_train_steps * hparams["warmup_ratio"])

        lr_scheduler = get_scheduler(
            name=hparams["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        # ==========================
        #  学習ループ
        # ==========================
        best_eval_loss   = float("inf")
        patience_counter = 0
        early_stop       = False
        pruned           = False

        # save_model=True のときだけ履歴と図を残す
        train_loss_history = []
        eval_loss_history  = []

        for epoch in range(num_epochs):
            # ---- Train ----
            model.train()
            train_sampler.set_epoch(epoch)

            total_train_loss = 0.0
            num_train_steps  = 0

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
                num_train_steps  += 1

            # rankごとの平均 train loss
            local_avg_train_loss = total_train_loss / max(1, num_train_steps)

            # train loss を all_reduce して全GPU平均
            train_loss_tensor = torch.tensor(local_avg_train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (train_loss_tensor / world_size).item()

            # ---- Eval (全rankで実行) ----
            model.eval()
            total_eval_loss = 0.0
            num_eval_steps  = 0

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
                    num_eval_steps  += 1

            # rankごとの eval loss 合計とステップ数を all_reduce
            loss_tensor  = torch.tensor(total_eval_loss, device=device)
            steps_tensor = torch.tensor(float(num_eval_steps), device=device)

            dist.all_reduce(loss_tensor,  op=dist.ReduceOp.SUM)
            dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)

            avg_eval_loss = (loss_tensor / steps_tensor).item()

            # ---- rank0 で early stopping & pruner 判定 & ログ ----
            prune_flag = False
            if rank == 0:
                print(
                    f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, "
                    f"Eval Loss: {avg_eval_loss:.4f}"
                )

                if save_model:
                    train_loss_history.append(avg_train_loss)
                    eval_loss_history.append(avg_eval_loss)

                # early stopping（ベスト更新時のみ保存）
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss   = avg_eval_loss
                    patience_counter = 0

                    if save_model:
                        save_dir = hparams["save_dir"]
                        os.makedirs(save_dir, exist_ok=True)

                        # ベストモデル保存
                        model.module.save_pretrained(os.path.join(save_dir, "best_model"))
                        tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

                        # loss log 保存
                        pd.DataFrame(
                            {
                                "epoch": list(range(1, len(train_loss_history) + 1)),
                                "train_loss": train_loss_history,
                                "eval_loss":  eval_loss_history,
                            }
                        ).to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)

                        # loss curve プロット
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
                        plt.title("Pretraining Loss Curve")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
                        plt.close()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("[Rank 0] Early stopping triggered")
                        early_stop = True

                # ---- Optuna pruner: trial が入っている場合のみ ----
                if ("trial" in hparams) and (not save_model):
                    trial = hparams["trial"]
                    trial.report(avg_eval_loss, step=epoch)
                    if trial.should_prune():
                        print(f"[Rank 0] Trial pruned at epoch {epoch+1}")
                        prune_flag = True

            # early_stop / prune フラグを全rankに共有
            early_stop_tensor = torch.tensor(1 if early_stop else 0, device=device)
            prune_tensor      = torch.tensor(1 if prune_flag else 0, device=device)

            dist.broadcast(early_stop_tensor, src=0)
            dist.broadcast(prune_tensor,      src=0)

            early_stop = bool(early_stop_tensor.item())
            prune_flag = bool(prune_tensor.item())

            # prune 優先
            if prune_flag:
                pruned = True
                if rank == 0:
                    # ここでは直近の eval loss を返しておく（Optuna側で TrialPruned を投げる）
                    return_dict["best_eval_loss"] = float(avg_eval_loss)
                    return_dict["pruned"] = True
                dist.destroy_process_group()
                return

            if early_stop:
                break

        # rank0 から結果を親プロセスへ返す
        if rank == 0 and not pruned:
            return_dict["best_eval_loss"] = float(best_eval_loss)
            return_dict["pruned"] = False

        dist.destroy_process_group()

    except Exception as e:
        print(f"\n[Rank {rank}] エラー発生: {e}")
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        if rank == 0:
            return_dict["best_eval_loss"] = float("inf")
            return_dict["pruned"] = False


# ===============================
#  Optuna 用ハイパーパラ構築（BERT-base 構造は固定）
# ===============================
def build_hparams_from_trial(trial: optuna.trial.Trial) -> dict:
    hparams = {}

    # 学習率・正則化
    hparams["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    hparams["weight_decay"]  = trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True)

    # warmup割合
    hparams["warmup_ratio"]  = trial.suggest_float("warmup_ratio", 0.01, 0.15)

    # dropout 系
    hparams["hidden_dropout"]    = trial.suggest_float("hidden_dropout",    0.0, 0.3)
    hparams["attention_dropout"] = trial.suggest_float("attention_dropout", 0.0, 0.2)

    # スケジューラ
    hparams["lr_scheduler_type"] = trial.suggest_categorical(
        "lr_scheduler_type", ["linear", "cosine"]
    )

    # BERT-base 構造は固定（hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072）
    # → ここでは探索しない

    # 学習エポックと early stopping 用 patience
    hparams["patience"]   = trial.suggest_int("patience",   2, 4)

    return hparams


# ===============================
#  Optuna Objective
# ===============================
def objective(trial: optuna.trial.Trial) -> float:
    # ハイパーパラ取得（構造は固定）
    hparams = build_hparams_from_trial(trial)

    # 探索フェーズでは save_dir は使わないが、キーは用意しておく
    hparams["save_dir"] = "dummy_dir"

    # trial を hparams に埋め込んで pruner から参照できるようにする
    hparams["trial"] = trial

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("CUDA GPU が見つからない")

    with Manager() as manager:
        return_dict = manager.dict()

        # 探索中は save_model=False で呼ぶ → モデルは保存しない
        mp.spawn(
            ddp_worker,
            args=(world_size, hparams, return_dict, False),
            nprocs=world_size,
            join=True,
        )

        best_eval_loss = float(return_dict.get("best_eval_loss", float("inf")))
        pruned = bool(return_dict.get("pruned", False))

    if pruned:
        # pruned trial として Optuna 側に伝える
        raise optuna.TrialPruned()

    return best_eval_loss


# ===============================
#  エントリーポイント
# ===============================
if __name__ == "__main__":
    # DDP 用環境変数（ローカルマシン想定）
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_DEBUG"] = "WARN"

    set_seed(42)

    # ===== Optuna で探索（prunerつき） =====
    study = optuna.create_study(
        study_name='phase_1',
        direction="minimize",
        storage="sqlite:///optuna_phase1.db",
        load_if_exists=True,
        pruner=MedianPruner(n_warmup_steps=2),
    )

    # 試行回数は環境に応じて調整（ここでは例として 15）
    study.optimize(objective, n_trials=5)

    print("========== Optuna Result ==========")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best eval loss   : {study.best_trial.value}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # ===== ベスト trial のパラメータでもう一度だけ学習し、ここで初めて保存 =====
    best_trial = study.best_trial

    # best_trial.params から hparams を再構成するための簡易ラッパ
    class DummyTrial:
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

    best_hparams["save_dir"] = "pretraining_bert_best"
    # 最終学習では pruner を使わないので trial キーは持たせない

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("CUDA GPU が見つからない")

    print("===== Training final model with best hyperparameters and saving it =====")

    with Manager() as manager:
        return_dict = manager.dict()
        mp.spawn(
            ddp_worker,
            args=(world_size, best_hparams, return_dict, True),
            nprocs=world_size,
            join=True,
        )

    print("Final model saved in: pretraining_bert_best/")

#{'learning_rate': 0.00016707717053270527, 'weight_decay': 0.0014551057527864568, 'warmup_ratio': 0.022546325260220136, 'batch_size': 64, 'hidden_dropout': 0.1323964613567746, 'attention_dropout': 0.1118470535270139, 'lr_scheduler_type': 'linear', 'num_layers': 12, 'hidden_size': 768, 'num_heads': 12, 'intermediate_size': 3072, 'num_epochs': 6, 'patience': 2}