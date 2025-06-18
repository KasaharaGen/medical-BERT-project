import gc
import torch
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, BertForSequenceClassification

# === データ読み込み ===
ehr_df = pd.read_csv("ehr_dengue_data.csv")
texts = ehr_df["text"].tolist()
labels = ehr_df["label"].astype(int).tolist()

# === データ分割 ===
texts_train, texts_tmp, labels_train, labels_tmp = train_test_split(texts, labels, test_size=0.3, random_state=42)
texts_val, texts_test, labels_val, labels_test = train_test_split(texts_tmp, labels_tmp, test_size=0.5, random_state=42)

# === トークナイザーと事前学習済みモデルの読み込み ===
tokenizer = PreTrainedTokenizerFast.from_pretrained("pretrain_phase2_tokenizer")
base_config = BertConfig.from_pretrained("pretrain_phase2_model")
base_config.num_labels = 2
pretrained_mlm = BertForMaskedLM.from_pretrained("pretrain_phase2_model")

# === Dataset定義 ===
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ClassificationDataset(texts_train, labels_train, tokenizer)
val_dataset = ClassificationDataset(texts_val, labels_val, tokenizer)
test_dataset = ClassificationDataset(texts_test, labels_test, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Optuna最適化対象関数 ===
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 2, 5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification(base_config)
    model.bert.load_state_dict(pretrained_mlm.bert.state_dict())
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    mcc = matthews_corrcoef(all_labels, all_preds)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return mcc

# === Optuna実行 ===
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_params = study.best_params

# === 最良設定で再学習・評価 ===
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

model = BertForSequenceClassification(base_config)
model.bert.load_state_dict(pretrained_mlm.bert.state_dict())
model.to(device)
optimizer = AdamW(model.parameters(), lr=best_params["learning_rate"])

train_losses = []
for epoch in range(best_params["epochs"]):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

# === テスト評価 ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

test_mcc = matthews_corrcoef(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
print(f"Test MCC: {test_mcc:.4f}")

# === 学習曲線 ===
plt.figure()
plt.plot(range(1, best_params["epochs"] + 1), train_losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("finetune_training_loss.png")

# === 混同行列 ===
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred:Neg", "Pred:Pos"], yticklabels=["Actual:Neg", "Actual:Pos"])
plt.title("Confusion Matrix (Dengue Detection)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("finetune_confusion_matrix.png")
plt.show()
