import torch
import torch.nn as nn
import optuna
from sklearn.metrics import matthews_corrcoef

class BaseOptimizeDNN:
    def __init__(self, model, input_dim, patience=10, num_epochs=100):
        self.model = model
        self.input_dim = input_dim
        self.patience = patience
        self.num_epochs = num_epochs

    def train_and_evaluate(self, trial, train_loader, val_loader, device):
        # 共通ハイパーパラメータのサンプリング
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
        
        # モデルの初期化
        model = self.model(input_dim=self.input_dim, dropout=dropout).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Early Stoppingの初期化
        best_val_loss = float('inf')
        patience_counter = 0 

        for epoch in range(self.num_epochs):
            # 訓練フェーズ
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # バリデーションフェーズ
            model.eval()
            val_loss, val_true, val_pred = self.validate(model, val_loader, criterion, device)
            
            # Early Stopping判定
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Optunaのプルーニング
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 最終評価指標
        mcc = matthews_corrcoef(val_true, val_pred)
        return mcc

    @staticmethod
    def validate(model, val_loader, criterion, device):
        val_loss = 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val).squeeze()
                val_loss += criterion(val_outputs, y_val).item()
                val_true.extend(y_val.cpu().numpy())
                val_pred.extend((val_outputs >= 0.5).cpu().numpy())
        val_loss /= len(val_loader)
        return val_loss, val_true, val_pred


class OptimizeFTTransformer:
    def __init__(self,FTTtransformer,input_dim,patience,num_epochs):
        self.model = FTTtransformer
        self.input_dim = input_dim
        self.patience = patience
        self.num_epochs = num_epochs
        return

    def objective(self,trial,train_loader,val_loader):
        # ハイパーパラメータのサンプリング
        embed_dim=trial.suggest_categorical("embed_dim",[16,32,64,128,256])
        num_heads=trial.suggest_categorical("num_heads",[4,8,16])
        num_layers=trial.suggest_categorical("num_layers",[3,4,5])
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)  
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)   

        # モデルの定義
        model = self.model(
            input_dim=self.inpput_dim,
            num_classes=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Early Stoppingの設定
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0

        # 訓練ループ
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch.squeeze())
                loss.backward()
                optimizer.step()

            # バリデーション評価
            model.eval()
            val_loss = 0
            val_true, val_pred = [], []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = model(X_val).squeeze()
                    val_loss += criterion(val_outputs, y_val.squeeze()).item()
                    predictions = (val_outputs >=0.5).float()
                    val_true.extend(y_val.cpu().numpy())
                    val_pred.extend(predictions.cpu().numpy())

            val_loss /= len(val_loader)

            # Early Stoppingの判定
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Optunaのプルーニング機能
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mcc = matthews_corrcoef(val_true, val_pred)
        print(f"Final MCC: {mcc:.4f}")

        return mcc  
    
class OptimizeResNet:
    def __init__(self,input_dim,ResNetBinaryClassifier,patience,num_epochs,train_loader,val_loader):
        self.input_dim=input_dim
        self.model=ResNetBinaryClassifier
        self.patience = patience
        self.num_epochs = num_epochs
        self.train_loader=train_loader
        self.val_loader=val_loader
        return
    
    def objective(self,trial):
        # ハイパーパラメータのサンプリング
        hidden_dim=trial.suggest_categorical("hidden_dim",[64,128,256])
        num_blocks=trial.suggest_categorical("num_blocks",[3,4,5])
        dropout1 = trial.suggest_float("dropout1", 0, 0.5, step=0.05)
        dropout2 = trial.suggest_float("dropout2", 0, 0.5, step=0.05)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)  
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)   

        # モデルの定義
        model = self.model(
            input_dim=self.input_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dropout1=dropout1,
            dropout2=dropout2
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Early Stoppingの設定
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0

        # 訓練ループ
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch.squeeze())
                loss.backward()
                optimizer.step()

            # バリデーション評価
            model.eval()
            val_loss = 0
            val_true, val_pred = [], []
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = model(X_val).squeeze()
                    val_loss += criterion(val_outputs, y_val.squeeze()).item()
                    predictions = (val_outputs >=0.5).float()
                    val_true.extend(y_val.cpu().numpy())
                    val_pred.extend(predictions.cpu().numpy())

            val_loss /= len(self.val_loader)

            # Early Stoppingの判定
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Optunaのプルーニング機能
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mcc = matthews_corrcoef(val_true, val_pred)
        print(f"Final MCC: {mcc:.4f}")

        return mcc  
    
    