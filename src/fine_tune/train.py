import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np

# === データ読み込み ===
df = pd.read_csv("processed_classification_dataset.csv")

# === テキスト・ラベル抽出 ===
texts = df["cleaned_text"].tolist()
labels = df["label"].tolist()

# === train/valid 分割 ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# === トークナイザー読み込み（自作なら "tokenizer" フォルダ）===
tokenizer = BertTokenizerFast.from_pretrained("../pretraining_bert/tokenizer")

# === Dataset クラス定義 ===
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# === Dataset の生成 ===
train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)

# === クラス重み（pos_weight）計算 ===
class_counts = np.bincount([int(x) for x in train_labels])
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float)

# === モデル定義（Binary classification, pos_weight付き）===
model = BertForSequenceClassification.from_pretrained("../pretraining_bert/output/pretrained_bert", num_labels=1)
model.classifier = torch.nn.Linear(model.config.hidden_size, 1)
model.config.problem_type = "single_label_classification"
model.config.num_labels = 1

# BCEWithLogitsLoss + pos_weight
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits.view(-1)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
    loss = loss_fct(logits, labels.float())
    return (loss, outputs) if return_outputs else loss

# === Training 設定 ===
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# === Trainer の定義 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_loss=compute_loss  # 上で定義した関数
)

# === 学習開始 ===
trainer.train()

# === モデル保存 ===
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("tokenizer")
