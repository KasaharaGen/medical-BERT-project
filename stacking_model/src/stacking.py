import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import matthews_corrcoef
import numpy as np
import optuna

class OutputBaseModel:
    def __init__(self,model_paths,model_class,input_dim,X_train_tensor):
        self.model_paths=model_paths
        self.model_class=model_class
        self.input_dim=input_dim
        self.X_train_tensor=X_train_tensor
        return
    
    def output_base_model(self):
        models=[]

        for i, path in enumerate(self.model_paths):
            if i in self.model_class:
                model = self.model_class[i](self.input_dim).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                models.append(model)
        train_outputs = []
        test_outputs = []

        with torch.no_grad():
            for model in models:
                model.eval()
                
                train_output = model(self.X_train_tensor.to(device))
                train_outputs.append(train_output)
                
                # テストデータの出力
                test_output = model(self.X_test_tensor.to(device))
                test_outputs.append(test_output)

        train_DL_features = torch.cat(train_outputs, dim=1)  
        test_DL_features = torch.cat(test_outputs, dim=1) 

        # 出力形状の確認
        print(f"Train DL Features Shape: {train_DL_features.shape}")
        print(f"Test DL Features Shape: {test_DL_features.shape}")



class StackingFeatureEngineering:
    """特徴量エンジニアリングを行うクラス"""

    def __init__(self, n_clusters=6, pca_components=6, tsne_components=3, random_state=0):
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.tsne_components = tsne_components
        self.random_state = random_state

    def normalize(self, data):
        """データの正規化"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, data.shape[-1]))

    def apply_kmeans(self, data):
        """KMeansクラスタリング"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return kmeans.fit_predict(data).reshape(-1, 1)

    def apply_pca(self, data):
        """PCAによる次元削減"""
        pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        return pca.fit_transform(data)

    def apply_tsne(self, data):
        """t-SNEによる次元削減"""
        tsne = TSNE(n_components=self.tsne_components, random_state=self.random_state)
        return tsne.fit_transform(data)

    def calculate_similarity(self, data):
        """データの類似度を計算"""
        distance_matrix = cdist(data, data, metric='euclidean')
        similarity_features = 1 / (1 + distance_matrix)
        return similarity_features.mean(axis=1).reshape(-1, 1)

    def process(self, data):
        """すべての特徴量エンジニアリングを適用"""
        normalized_data = self.normalize(data)
        cluster_features = self.apply_kmeans(normalized_data)
        pca_features = self.apply_pca(normalized_data)
        tsne_features = self.apply_tsne(normalized_data)
        similarity_features = self.calculate_similarity(normalized_data)

        combined_features = np.hstack([normalized_data, cluster_features, pca_features, tsne_features, similarity_features])
        return StandardScaler().fit_transform(combined_features)


class MetaModel(nn.Module):
    """スタッキングモデルのメタモデル"""

    def __init__(self, input_dim, num_classes, embed_dim, num_heads, num_layers, dropout):
        super(MetaModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_dropout(x)
        return torch.sigmoid(self.fc(x))


class OptimizeStackingModel:
    """Optunaを用いてメタモデルを最適化"""

    def __init__(self, train_loader, val_loader, input_dim):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim

    def objective(self, trial):
        # ハイパーパラメータのサンプリング
        embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        num_layers = trial.suggest_categorical("num_layers", [3, 4, 5])
        dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)

        # モデルの初期化
        model = MetaModel(input_dim=self.input_dim, num_classes=1, embed_dim=embed_dim,
                          num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Early Stoppingの設定
        patience, best_val_loss, patience_counter = 10, float('inf'), 0

        # トレーニングループ
        for epoch in range(100):
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
            val_loss, val_true, val_pred = 0, [], []
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = model(X_val).squeeze()
                    val_loss += criterion(val_outputs, y_val.squeeze()).item()
                    predictions = (val_outputs >= 0.5).float()
                    val_true.extend(y_val.cpu().numpy())
                    val_pred.extend(predictions.cpu().numpy())

            val_loss /= len(self.val_loader)

            # Early Stoppingの判定
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Optunaのプルーニング機能
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 評価指標
        return matthews_corrcoef(val_true, val_pred)
