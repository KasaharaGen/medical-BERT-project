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
#  モデル微調整ユーティリティ（LLRD / 部分凍結 / LoRA任意）
# ===============================
def freeze_lower_layers(model, n_freeze: int = 0, freeze_embeddings: bool = False):
    """
    下位 n_freeze 層と必要に応じて埋め込み層を凍結する。
    """
    if freeze_embeddings:
        for p in model.bert.embeddings.parameters():
            p.requires_grad = False
    if n_freeze > 0:
        encoder_layers = model.bert.encoder.layer
        for i in range(n_freeze):
            for p in encoder_layers[i].parameters():
                p.requires_grad = False


def build_llrd_param_groups(model, base_lr: float, weight_decay: float,
                            llrd_decay: float = 0.95):
    """
    LLRD: 下位層ほど学習率を小さくする（層ごとに lr *= llrd_decay）。
    - 重複登録を防ぐため seen セットで管理
    - decoder.weight（埋め込みと weight tying）は LM ヘッド側からは登録しない
    """
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = []
    seen = set()  # すでに登録したパラメータ（id）を記録

    def add_group(params_list, wd, lr):
        params = [p for p in params_list if p.requires_grad and (id(p) not in seen)]
        if params:
            for p in params:
                seen.add(id(p))
            param_groups.append({"params": params, "weight_decay": wd, "lr": lr})

    # --- Embeddings ---
    lr = base_lr * (llrd_decay ** 13)
    emb_decay, emb_nodecay = [], []
    for n, p in model.bert.embeddings.named_parameters():
        (emb_nodecay if any(nd in n for nd in no_decay) else emb_decay).append(p)
    add_group(emb_decay,    weight_decay, lr)
    add_group(emb_nodecay,  0.0,          lr)

    # --- Encoder layers (0..11) ---
    for layer_idx in range(12):
        layer = model.bert.encoder.layer[layer_idx]
        lr = base_lr * (llrd_decay ** (12 - layer_idx))  # 上層ほど lr 大
        decay_params, nodecay_params = [], []
        for n, p in layer.named_parameters():
            (nodecay_params if any(nd in n for nd in no_decay) else decay_params).append(p)
        add_group(decay_params,   weight_decay, lr)
        add_group(nodecay_params, 0.0,          lr)

    # --- Pooler（存在時のみ） ---
    pooler = getattr(model.bert, "pooler", None)
    head_decay, head_nodecay = [], []
    head_lr = base_lr
    if pooler is not None:
        for n, p in pooler.named_parameters():
            (head_nodecay if any(nd in n for nd in no_decay) else head_decay).append(p)

    # --- LM Head（decoder.weight は weight tying のためスキップ） ---
    for n, p in model.cls.named_parameters():
        if n.endswith("decoder.weight"):
            continue  # embeddings と同一テンソルなので除外
        (head_nodecay if any(nd in n for nd in no_decay) else head_decay).append(p)

    add_group(head_decay,   weight_decay, head_lr)
    add_group(head_nodecay, 0.0,          head_lr)

    return param_groups


def maybe_apply_lora(model, r=8, alpha=16, dropout=0.05, target_modules=("query", "key", "value", "dense")):
    """
    LoRA/Adapter を任意で挿入（peft がある場合のみ）。ない場合はそのまま返す。
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        return model, False  # PEFT未導入

    lora_targets = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            lora_targets.append(name)

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # MLMでも可動（分類タスク種別は学習自体には大勢に影響なし）
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(set(target_modules)),
    )
    model = get_peft_model(model, lora_config)
    return model, True


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
        # Phase1保存構造（best_model / tokenizer）想定：tokenizerは tokenizer/ から読む
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "../pretraining_bert_1/pretraining_bert_best/tokenizer"
        )

        # === Dataset / DataLoader ===
        max_length = 512
        mlm_probability = 0.15
        batch_size = 32

        train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
        eval_dataset  = TextDataset(eval_texts,  tokenizer, max_length=max_length)

        #train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
        #eval_dataset  = Subset(eval_dataset,  range(min(1000, len(eval_dataset))))

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset,  num_replicas=world_size, rank=rank, shuffle=False
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

        # === モデルロード（BERT-base 構造固定） ===
        base_model_dir = "../pretraining_bert_1/pretraining_bert_best/best_model"
        config = BertConfig.from_pretrained(base_model_dir)
        # 構造を明示固定（念のため上書き）
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        config.intermediate_size = 3072
        # dropout は探索パラメータで上書き
        config.hidden_dropout_prob = hparams["hidden_dropout"]
        config.attention_probs_dropout_prob = hparams["attention_dropout"]

        model = BertForMaskedLM.from_pretrained(base_model_dir, config=config).to(device)

        # === 部分凍結（先頭/末尾の層を固定したい場合） ===
        # 例：下位 n 層を凍結（中間破壊の抑制）。必要なときだけ値を >0 にする
        n_freeze_lower = hparams.get("freeze_lower_n", 0)          # 推奨デフォルト: 0
        freeze_embeddings = hparams.get("freeze_embeddings", False) # 推奨デフォルト: False
        freeze_lower_layers(model, n_freeze_lower, freeze_embeddings)

        # === 任意：LoRA/Adapter の導入（peft があれば有効化） ===
        if hparams.get("use_lora", False):
            model, lora_on = maybe_apply_lora(
                model,
                r=hparams.get("lora_r", 8),
                alpha=hparams.get("lora_alpha", 16),
                dropout=hparams.get("lora_dropout", 0.05),
                target_modules=hparams.get("lora_targets", ("query", "key", "value", "dense")),
            )
            if (rank == 0) and (not lora_on):
                print("[Phase2] PEFT未導入のためLoRAはスキップした")

        model = DDP(model, device_ids=[rank])

        # === Optimizer & Scheduler （LLRD対応） ===
        # LLRDの有無を切替
        use_llrd = hparams.get("use_llrd", True)
        if use_llrd:
            param_groups = build_llrd_param_groups(
                model.module,
                base_lr=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"],
                llrd_decay=hparams.get("llrd_decay", 0.95),
            )
            optimizer = AdamW(param_groups)
        else:
            # 従来の一括LR
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": hparams["weight_decay"],
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=hparams["learning_rate"])

        num_epochs = 5
        patience   = hparams["patience"]
        min_delta  = hparams["min_delta"]

        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps    = int(max_train_steps * hparams["warmup_ratio"])

        lr_scheduler = get_scheduler(
            name=hparams["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        best_eval_loss    = float("inf")
        epochs_no_improve = 0
        early_stop        = False
        pruned            = False

        train_loss_history = []
        eval_loss_history  = []

        # === 学習ループ ===
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

            local_avg_train_loss = total_train_loss / max(1, num_train_steps)

            # 全GPU平均
            train_loss_tensor = torch.tensor(local_avg_train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (train_loss_tensor / world_size).item()

            # ---- Eval ----
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

            loss_tensor  = torch.tensor(total_eval_loss, device=device)
            steps_tensor = torch.tensor(float(num_eval_steps), device=device)

            dist.all_reduce(loss_tensor,  op=dist.ReduceOp.SUM)
            dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)

            avg_eval_loss = (loss_tensor / steps_tensor).item()

            # ---- rank0: early stopping & pruner & ログ ----
            prune_flag = False
            if rank == 0:
                print(
                    f"[Phase2] Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, "
                    f"Eval Loss: {avg_eval_loss:.6f}"
                )

                if save_model:
                    train_loss_history.append(avg_train_loss)
                    eval_loss_history.append(avg_eval_loss)

                # Early Stopping
                if avg_eval_loss < best_eval_loss - min_delta:
                    best_eval_loss    = avg_eval_loss
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

            # ブロードキャスト
            early_stop_tensor = torch.tensor(1 if early_stop else 0, device=device)
            prune_tensor      = torch.tensor(1 if prune_flag else 0, device=device)

            dist.broadcast(early_stop_tensor, src=0)
            dist.broadcast(prune_tensor, src=0)

            early_stop = bool(early_stop_tensor.item())
            prune_flag = bool(prune_tensor.item())

            if prune_flag:
                pruned = True
                if rank == 0:
                    return_dict["best_eval_loss"] = float(avg_eval_loss)
                    return_dict["pruned"] = True
                dist.destroy_process_group()
                return

            if early_stop:
                break

        # rank0 戻り値／保存
        if rank == 0 and not pruned:
            return_dict["best_eval_loss"] = float(best_eval_loss)
            return_dict["pruned"] = False

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

                # loss ログ
                pd.DataFrame(
                    {
                        "epoch": list(range(1, len(train_loss_history) + 1)),
                        "train_loss": train_loss_history,
                        "eval_loss":  eval_loss_history,
                    }
                ).to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)

                # loss curve
                plt.figure(figsize=(8, 5))
                plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
                plt.plot(range(1, len(eval_loss_history) + 1),  eval_loss_history,  label="Eval Loss")
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
#  Optuna 用ハイパーパラ構築（Phase2推奨レンジ）
# ===============================
def build_hparams_from_trial(trial: optuna.trial.Trial) -> dict:
    hparams = {}

    # === 探索推奨（Phase2） ===
    hparams["learning_rate"]     = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    hparams["weight_decay"]      = trial.suggest_float("weight_decay",  1e-6, 5e-3, log=True)
    hparams["warmup_ratio"]      = trial.suggest_float("warmup_ratio",  0.0, 0.05)
    hparams["hidden_dropout"]    = trial.suggest_float("hidden_dropout",    0.1, 0.4)
    hparams["attention_dropout"] = trial.suggest_float("attention_dropout", 0.1, 0.3)
    hparams["lr_scheduler_type"] = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"])
    hparams["patience"]          = trial.suggest_int("patience",   2, 3)
    hparams["min_delta"]         = trial.suggest_float("min_delta", 1e-5, 5e-4, log=True)

    # === 構造固定（BERT-base）なのでここでは触らない ===
    # hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072

    # === 追加テク：LLRD / 凍結 / LoRA（任意で固定or探索も可） ===
    hparams["use_llrd"]          = True
    hparams["llrd_decay"]        = trial.suggest_float("llrd_decay", 0.90, 0.97)  # 上層ほど高LR
    hparams["freeze_lower_n"]    = trial.suggest_int("freeze_lower_n", 0, 4)      # 下位0〜4層を凍結
    hparams["freeze_embeddings"] = trial.suggest_categorical("freeze_embeddings", [False, True])

    # LoRA（PEFTがあれば使用、なければ自動スキップ）
    hparams["use_lora"]      = trial.suggest_categorical("use_lora", [False, True])
    hparams["lora_r"]        = 8
    hparams["lora_alpha"]    = 16
    hparams["lora_dropout"]  = 0.05
    hparams["lora_targets"]  = ("query", "key", "value", "dense")

    return hparams


# ===============================
#  Optuna Objective
# ===============================
def objective(trial: optuna.trial.Trial) -> float:
    hparams = build_hparams_from_trial(trial)

    # save_dir は探索中は未使用
    hparams["save_dir"] = "dummy_dir"
    # pruner用に trial を埋め込み
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
        study_name = 'phase_2',
        direction="minimize",
        pruner=MedianPruner(n_warmup_steps=1),
        storage="sqlite:///optuna_phase2.db",
        load_if_exists=True    
    )

    '''study.optimize(objective, n_trials=5)

    print("========== Phase2 Optuna Result ==========")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best eval loss   : {study.best_trial.value}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")'''

    # ===== ベスト trial で再学習し保存（Phase1と同じ構造：best_model/tokenizer/ログ） =====
    best_trial = study.best_trial

    class DummyTrial:
        def __init__(self, params): self.params = params
        def suggest_float(self, name, low, high, log=False):   return float(self.params[name])
        def suggest_categorical(self, name, choices):           return self.params[name]
        def suggest_int(self, name, low, high):                 return int(self.params[name])

    dummy_trial  = DummyTrial(best_trial.params)
    best_hparams = build_hparams_from_trial(dummy_trial)
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
