import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm

# === モデル・トークナイザーの読み込み ===
model = BertForMaskedLM.from_pretrained("../pretraining_bert_1/pretrain_phase1_model")
tokenizer = PreTrainedTokenizerFast.from_pretrained("../pretraining_bert_1/pretrain_phase1_tokenizer")

# === GPU設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# === データ読み込み（デング熱アブストラクト） ===
df = pd.read_csv("dengue_abstracts.csv")
dengue_texts = df["abstract"].dropna().astype(str).tolist()

# 訓練/検証データに分割
train_ratio = 0.98
train_size = int(len(dengue_texts) * train_ratio)
train_texts = dengue_texts[:train_size]
eval_texts = dengue_texts[train_size:]

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

train_dataset = TextDataset(train_texts, tokenizer)
eval_dataset = TextDataset(eval_texts, tokenizer)

# === Collator・DataLoader定義 ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=data_collator)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, collate_fn=data_collator)

# === オプティマイザ（weight_decayあり） ===
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

# === 学習ループ ===
num_epochs = 2
train_losses_phase2 = []
eval_losses_phase2 = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
    for batch in loop:
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_eval_loss = 0.0
    eval_loop = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Eval]", leave=False)
    with torch.no_grad():
        for batch in eval_loop:
            batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            total_eval_loss += loss.item()
            eval_loop.set_postfix(eval_loss=loss.item())
    avg_eval_loss = total_eval_loss / len(eval_loader)

    train_losses_phase2.append(avg_train_loss)
    eval_losses_phase2.append(avg_eval_loss)
    print(f"[Phase2] Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

# === 学習曲線の可視化 ===
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses_phase2, label='Train Loss')
plt.plot(epochs, eval_losses_phase2, label='Eval Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pretraining Phase 2 Loss Curve (Dengue Corpus)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pretrain_phase2_loss_curve.png")
plt.show()

# === モデルとトークナイザーの保存 ===
model_to_save = model.module if hasattr(model, "module") else model
model_to_save.save_pretrained("pretrain_phase2_model")
tokenizer.save_pretrained("pretrain_phase2_tokenizer")
