import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class PreprocessText:
    """PDFからテキストを抽出し、特定の単語を判定するクラス"""

    @staticmethod
    def ocr(pdf_path, dpi=300):
        """PDFファイルをOCR処理し、テキストを抽出する"""
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            text = ""

            for i, image in enumerate(images):
                temp_image_path = f"temp_page_{i}.png"
                image.save(temp_image_path, "PNG")
                text += pytesseract.image_to_string(Image.open(temp_image_path), lang="eng")
                os.remove(temp_image_path)

            # 特殊文字を削除して正規化
            text = text.lower()
            delete_chars = {char: None for char in '.,:;"!#$%&\'()*=~{}[]/-_^@'}
            text = text.translate(str.maketrans(delete_chars))
            return " ".join(text.split())
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return None

    @staticmethod
    def extract_words_to_dict(text, extraction_words):
        """特定の単語がテキスト内に存在するかを判定"""
        return {word: int(word in text) for word in extraction_words}


class FeatureEngineering:
    """データに対する特徴量エンジニアリングを行うクラス"""

    def __init__(self, random_state=0):
        self.random_state = random_state

    @staticmethod
    def normalize_data(df, columns, scaler=MinMaxScaler):
        """データを正規化"""
        transformer = Pipeline(steps=[("scaler", scaler())])
        preprocessor = ColumnTransformer(transformers=[("num", transformer, columns)])
        normalized_data = preprocessor.fit_transform(df)
        return pd.DataFrame(normalized_data, columns=columns)

    def apply_pca(self, df, n_components=10):
        """PCAによる次元削減"""
        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pd.DataFrame(pca.fit_transform(df), columns=[f"pca_{i+1}" for i in range(n_components)])

    def apply_tsne(self, df, n_components=3, perplexity=30):
        """t-SNEによる次元削減"""
        tsne = TSNE(n_components=n_components, random_state=self.random_state, perplexity=perplexity)
        return pd.DataFrame(tsne.fit_transform(df), columns=[f"tsne_{i+1}" for i in range(n_components)])

    def apply_clustering(self, df, n_clusters=10):
        """クラスタリングで特徴量を生成"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        kmeans.fit(df)
        return pd.DataFrame({
            "kmeans_dist": kmeans.transform(df).min(axis=1),
            "kmeans_cluster": kmeans.labels_
        })

    @staticmethod
    def calculate_similarity(df):
        """コサイン類似度を計算"""
        similarity = cosine_similarity(df)
        return pd.DataFrame({"nearest_cosine_similarity": similarity.mean(axis=1)})

    def feature_extract(self, df, label_column="dengue"):
        """特徴量エンジニアリングの統合処理"""
        label_df = df[label_column]
        df = df.drop(label_column, axis=1)

        # 正規化
        normalized_df = self.normalize_data(df, df.columns.tolist())

        # 特徴量生成
        features = pd.concat([
            self.apply_clustering(normalized_df),
            self.apply_pca(normalized_df),
            self.apply_tsne(normalized_df),
            self.calculate_similarity(normalized_df)
        ], axis=1)

        # 特徴量を正規化
        features = self.normalize_data(features, features.columns.tolist(), scaler=StandardScaler)

        # 元データと統合
        return pd.concat([df.reset_index(drop=True), features, label_df.reset_index(drop=True)], axis=1)


class LearningDataCreator:
    """学習用データを作成するクラス"""

    def __init__(self, learning_df, label_column="dengue"):
        self.df = learning_df
        self.label_column = label_column

    def create_learning_data(self, batch_size=128, test_size=0.2,val_size=0.1, random_state=42):
        """学習データを作成"""
        X = self.df.drop(self.label_column, axis=1).values
        y = self.df[self.label_column].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size,random_state=random_state)

        #torchテンソルに変換
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader,X_test_tensor,y_test_tensor
