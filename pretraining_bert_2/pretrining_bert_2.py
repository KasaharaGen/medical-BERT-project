import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt

# === CSVから文を読み込む ===
csv_path = "../infection_data/abstract_sentences.csv"  # sentence列を含むCSVファイル
df = pd.read_csv(csv_path)
sentences = df["sentence"].dropna().astype(str).tolist()  # 欠損除去 + 文字列変換

# データを訓練用と検証用に分割
train_ratio = 0.98
train_size = int(len(sentences) * train_ratio)
train_texts = sentences[:train_size]
eval_texts = sentences[train_size:]

# === トークナイザーの読み込み ===
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizer/tokenizer.json",
                                    unk_token="[UNK]", pad_token="[PAD]",
                                    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")

# === モデル設定 ===
vocab_size = tokenizer.vocab_size
config = BertConfig(vocab_size=vocab_size,
                    hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                    intermediate_size=3072, max_position_embeddings=512)
model = BertForMaskedLM(config)

# === GPU設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# === Dataset定義 ===
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"]}

# === データセットとデータローダー ===
train_dataset = TextDataset(train_texts, tokenizer)
eval_dataset = TextDataset(eval_texts, tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=data_collator)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, collate_fn=data_collator)

# === オプティマイザ ===
optimizer = AdamW(model.parameters(), lr=5e-5)

# === 学習ループ ===
num_epochs = 3
train_losses_phase1 = []
eval_losses_phase1 = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(eval_loader)

    train_losses_phase1.append(avg_train_loss)
    eval_losses_phase1.append(avg_eval_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

# 学習曲線の可視化
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses_phase1, label='Train Loss')
plt.plot(epochs, eval_losses_phase1, label='Eval Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pretraining Loss Curve (Phase 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pretrain_phase1_loss_curve.png")  # 画像として保存
plt.show()


# === モデルとトークナイザーの保存 ===
model_to_save = model.module if hasattr(model, "module") else model
model_to_save.save_pretrained("pretrain_bert_2_model")
tokenizer.save_pretrained("pretrain_bert_2_tokenizer")
