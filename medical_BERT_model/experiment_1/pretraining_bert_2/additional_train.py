import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

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

def ddp_main(rank, world_size, resume=False):
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        # === データの読み込み ===
        df = pd.read_csv("dengue_abstracts.csv")
        texts = df["abstract"].dropna().astype(str).tolist()
        split_idx = int(len(texts) * 0.98)
        train_texts, eval_texts = texts[:split_idx], texts[split_idx:]

        # === トークナイザーの読み込み ===
        tokenizer_path = "pretrain_phase2_tokenizer_ddp" if resume else "../pretraining_bert_1/pretrain_phase1_tokenizer"
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        train_dataset = TextDataset(train_texts, tokenizer)
        eval_dataset = TextDataset(eval_texts, tokenizer)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.2)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collator)
        eval_loader = DataLoader(eval_dataset, batch_size=32, sampler=eval_sampler, collate_fn=collator)

        # === モデルの読み込み ===
        model_path = "pretrain_phase2_model_ddp" if resume else "../pretraining_bert_1/pretrain_phase1_model"
        model = BertForMaskedLM.from_pretrained(model_path).to(rank)
        model = DDP(model, device_ids=[rank])

        # === Optimizer設定 ===
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

        # === 学習ループ ===
        num_epochs = 2
        train_losses, eval_losses = [], []

        for epoch in range(num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            total_train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1} Train", leave=False):
                batch = {k: torch.tensor(v).to(rank) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            if rank == 0:
                model.eval()
                total_eval_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc=f"[Rank {rank}] Eval", leave=False):
                        batch = {k: torch.tensor(v).to(rank) for k, v in batch.items()}
                        outputs = model(**batch)
                        total_eval_loss += outputs.loss.item()
                avg_eval_loss = total_eval_loss / len(eval_loader)
                train_losses.append(avg_train_loss)
                eval_losses.append(avg_eval_loss)
                print(f"[Phase2] Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

        # === 保存（rank 0 のみ） ===
        if rank == 0:
            model.module.save_pretrained("pretrain_phase2_model_ddp")
            tokenizer.save_pretrained("pretrain_phase2_tokenizer_ddp")

            pd.DataFrame({
                "epoch": list(range(1, num_epochs + 1)),
                "train_loss": train_losses,
                "eval_loss": eval_losses
            }).to_csv("loss_log_phase2_ddp.csv", index=False)

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
            plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Phase2 DDP Training Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("pretrain_phase2_ddp_loss_curve.png")
            plt.show()

        dist.destroy_process_group()

    except Exception as e:
        print(f"[Rank {rank}] エラー発生: {e}")
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except:
            pass
        exit(1)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_DEBUG"] = "INFO"

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(ddp_main, args=(world_size, True), nprocs=world_size, join=True)
