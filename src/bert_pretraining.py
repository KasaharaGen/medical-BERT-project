import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import os
from model import MyBERTConfig, MyBERTForMaskedLM
import sentencepiece as spm

# === 設定 ===
CSV_PATH = "../data/merged_blocks.csv"
CHECKPOINT_DIR = "./checkpoints/mybert"
SP_MODEL_PATH = "./tokenizer/mybert_tokenizer.model"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
MLM_PROB = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTIGPU = torch.cuda.device_count() > 1

# === データセット ===
class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        self.input_ids = self.inputs['input_ids']
        self.token_type_ids = self.inputs['token_type_ids'] if 'token_type_ids' in self.inputs else torch.zeros_like(self.input_ids)
        self.attention_mask = self.inputs['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, MLM_PROB)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        labels[~masked_indices] = -100
        return inputs, labels

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone()
        input_ids_masked, labels = self.mask_tokens(input_ids)
        return {
            'input_ids': input_ids_masked,
            'token_type_ids': self.token_type_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': labels
        }

# === データ準備 ===
df = pd.read_csv(CSV_PATH)
df = df[df['sentence'].notna() & (df['sentence'].str.strip() != "")]
texts = df['sentence'].tolist()
print(f"📚 データ読み込み完了: {len(texts)} 件")

# === 自作トークナイザ読み込み ===
tokenizer = PreTrainedTokenizerFast(tokenizer_file=SP_MODEL_PATH)
tokenizer.mask_token = "[MASK]"
tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"
tokenizer.sep_token = "[SEP]"
tokenizer.unk_token = "[UNK]"

# === モデル構築 ===
config = MyBERTConfig(vocab_size=tokenizer.vocab_size)
model = MyBERTForMaskedLM(config)
if MULTIGPU:
    model = nn.DataParallel(model)
model.to(DEVICE)

# === 学習ループ ===
dataset = MLMDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, tokenizer.vocab_size), batch['labels'].view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}: Loss = {avg:.4f}")

# === モデル保存 ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "mybert_mlm.pt"))
print(f"💾 モデル保存完了: {CHECKPOINT_DIR}/mybert_mlm.pt")