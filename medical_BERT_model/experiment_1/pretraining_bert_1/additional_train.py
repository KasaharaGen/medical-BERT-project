import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

# === Dataset定義 ===
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"]}

# === 各プロセスが実行する関数 ===
def main(rank, world_size, resume=False):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        # データ読み込み
        df = pd.read_csv("../infection_data/abstract_sentences.csv")
        sentences = df["sentence"].dropna().astype(str).tolist()
        train_size = int(len(sentences) * 0.98)
        train_texts = sentences[:train_size]
        eval_texts = sentences[train_size:]

        # トークナイザー
        tokenizer_dir = "pretrain_phase1_tokenizer_ddp" if resume else "../tokenizer"
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        # Dataset準備
        train_dataset = TextDataset(train_texts, tokenizer)
        eval_dataset = TextDataset(eval_texts, tokenizer)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.2)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collator)
        eval_loader = DataLoader(eval_dataset, batch_size=32, sampler=eval_sampler, collate_fn=collator)

        # モデルの読み込みまたは初期化
        if resume:
            model = BertForMaskedLM.from_pretrained("pretrain_phase1_model_ddp").to(rank)
        else:
            config = BertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                intermediate_size=3072, max_position_embeddings=512
            )
            model = BertForMaskedLM(config).to(rank)

        model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # エポック設定
        start_epoch = 0
        num_epochs = 3
        train_losses = []
        eval_losses = []

        if resume and rank == 0 and os.path.exists("loss_log.csv"):
            log_df = pd.read_csv("loss_log.csv")
            train_losses = log_df["train_loss"].tolist()
            eval_losses = log_df["eval_loss"].tolist()
            start_epoch = len(log_df)

        # 学習ループ
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            total_train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1} Train", leave=False):
                batch = {k: v.to(rank) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 評価（rank 0のみ）
            if rank == 0:
                model.eval()
                total_eval_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc=f"[Rank {rank}] Epoch {epoch+1} Eval", leave=False):
                        batch = {k: v.to(rank) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        total_eval_loss += loss.item()
                avg_eval_loss = total_eval_loss / len(eval_loader)
                eval_losses.append(avg_eval_loss)
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

        # モデル保存（rank 0）
        if rank == 0:
            model.module.save_pretrained("pretrain_phase1_model_ddp")
            tokenizer.save_pretrained("pretrain_phase1_tokenizer_ddp")

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
            plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Eval Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Pretraining Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("pretrain_loss_curve.png")
            plt.show()

            pd.DataFrame({
                "epoch": list(range(1, len(train_losses) + 1)),
                "train_loss": train_losses,
                "eval_loss": eval_losses
            }).to_csv("loss_log.csv", index=False)

        dist.destroy_process_group()

    except Exception as e:
        print(f"\n[Rank {rank}] エラー発生: {e}")
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except:
            pass
        exit(1)

# === 実行部分 ===
if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_DEBUG"] = "INFO"

    # Trueで再開学習、Falseで初期から
    torch.multiprocessing.spawn(main, args=(world_size, True), nprocs=world_size, join=True)
