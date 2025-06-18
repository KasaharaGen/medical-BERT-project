import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
from tqdm import tqdm

# Early Stopping 設定
early_stopping_rounds = 3
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
val_mccs = []

for epoch in range(best_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(best_train_loader, desc=f"Epoch {epoch+1}/{best_epochs}")
    for batch in pbar:
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
        pbar.set_postfix({"Train Loss": loss.item()})
    avg_train_loss = total_loss / len(best_train_loader)
    train_losses.append(avg_train_loss)

    # 検証
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in best_val_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    avg_val_loss = val_loss / len(best_val_loader)
    val_losses.append(avg_val_loss)

    val_mcc = matthews_corrcoef(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds)
    val_auc = roc_auc_score(all_labels, all_preds)
    val_mccs.append(val_mcc)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | MCC={val_mcc:.4f} | F1={val_f1:.4f} | AUC={val_auc:.4f}")

    # Early Stopping チェック
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # モデル保存
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), "best_finetuned_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_rounds:
            print("Early stopping triggered.")
            break
