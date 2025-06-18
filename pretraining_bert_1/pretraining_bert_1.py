import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CSVから文を読み込む ===
csv_path = "../infection_data/abstract_sentences.csv"
df = pd.read_csv(csv_path)
sentences = df["sentence"].dropna().astype(str).tolist()

# === データ分割 ===
train_ratio = 0.98
train_size = int(len(sentences) * train_ratio)
train_texts = sentences[:train_size]
eval_texts = sentences[train_size:]

# === トークナイザー ===
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="../tokenizer/tokenizer.json",
    unk_token="[UNK]", pad_token="[PAD]",
    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"
)

# === モデル設定 ===
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
    intermediate_size=3072, max_position_embeddings=512
)
model = BertForMaskedLM(config)

# === GPU設定（複数GPU対応） ===
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
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"]}

# === DataLoader ===
train_dataset = TextDataset(train_texts, tokenizer)
eval_dataset = TextDataset(eval_texts, tokenizer)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.2)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collator)
eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, collate_fn=collator)

# === Optimizer ===
optimizer = AdamW(model.parameters(), lr=5e-5)

# === 学習ループ ===
num_epochs = 3
train_losses, eval_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
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
        total_train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_train_loss = total_train_loss / len(train_loader)

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

    train_losses.append(avg_train_loss)
    eval_losses.append(avg_eval_loss)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

# === 学習曲線の可視化 ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), eval_losses, label='Eval Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pretraining Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pretrain_phase1_loss_curve.png")
plt.show()

# === モデル保存 ===
model_to_save = model.module if hasattr(model, "module") else model
model_to_save.save_pretrained("pretrain_phase1_model")
tokenizer.save_pretrained("pretrain_phase1_tokenizer")
