import numpy as np

import optuna

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression # ロジスティック回帰
from sklearn.neighbors import KNeighborsClassifier # K近傍法
from sklearn.svm import SVC # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier # 決定木
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from lightgbm import LGBMClassifier #LGBM
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
from sklearn.naive_bayes import GaussianNB # ナイーブ・ベイズ
from sklearn.ensemble import StackingClassifier


class optimize_function:
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        return

    def objective_logistic_regression(self,trial):
        # ソルバーの選択（L2正則化をサポートするソルバーのみ）
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs'])

        # 正則化はL2に固定
        penalty = 'l2'

        # Logistic Regressionモデルのインスタンス
        clf = LogisticRegression(
            penalty=penalty,
            C=trial.suggest_float('C', 0.0001, 100.0, log=True),
            solver=solver,
            max_iter=200
        )

        # 交差検証でスコアを計算
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()


    def objective_knn(self,trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 100)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])

        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()


    def objective_svm_rbf(self,trial):
        C = trial.suggest_float('C', 0.0001, 100.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        clf = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced', probability=True)

        y_prob = cross_val_predict(clf, self.X_train, self.y_train, cv=3, method='predict_proba')

        # しきい値0.7で予測
        threshold = 0.5
        y_pred = (y_prob[:, 1] >= threshold).astype(int)

        # 精度を計算
        return np.mean(y_pred == self.y_train)

    def objective_decision_tree(self,trial):
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 1, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)

        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, class_weight='balanced')
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()


    def objective_random_forest(self,trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, class_weight='balanced')
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()


    def objective_lgbm(self,trial):
        num_leaves = trial.suggest_int('num_leaves', 10, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        lambda_l1 = trial.suggest_float('lambda_l1', 0.0001, 100)
        lambda_l2 = trial.suggest_float('lambda_l2', 0.0001, 100)

        clf = LGBMClassifier(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
                             lambda_l1=lambda_l1, lambda_l2=lambda_l2, class_weight='balanced')
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()


    def objective_adaboost(self,trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 1.0)

        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()


    def objective_naive_bayes(self,trial):
        var_smoothing = trial.suggest_float('var_smoothing', 1e-9, 1e-7, log=True)

        clf = GaussianNB(var_smoothing=var_smoothing)
        score = cross_val_score(clf, self.X_train, self.y_train, cv=3)

        return score.mean()



class optimize():
    def __init__(self,objectives_dict):
        self.objectives_dict=objectives_dict
        print('コンストラクタが呼び出されました')
        return

    def optimize(self):
        best_params_dict={}

        for model_name, objective in self.objectives_dict.items():
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)

            best_params_dict[model_name] = study.best_params

        print(best_params_dict)

        return best_params_dict


class optimize_stacking():
    def __init__(self,names,classifiers,X_train,y_train):
        self.names = names
        self.classifiers = classifiers
        self.X_train = X_train
        self.y_train = y_train
        return


    def objective_stacking(self,trial):
        estimators = [(name, clf) for name, clf in zip(self.names, self.classifiers)]

        ##メタモデルによって適宜変更
        num_leaves = trial.suggest_int('num_leaves', 10, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        lambda_l1 = trial.suggest_float('lambda_l1', 0.0001, 100)
        lambda_l2 = trial.suggest_float('lambda_l2', 0.0001, 100)
        meta_model = LGBMClassifier(num_leaves=num_leaves,
                                    learning_rate=learning_rate,
                                    n_estimators=n_estimators,
                                    lambda_l1=lambda_l1,
                                    lambda_l2=lambda_l2,
                                    class_weight='balanced')

        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model)

        return cross_val_score(stacking_clf, self.X_train, self.y_train, cv=3).mean()


    def optimize_stacking(self,objective_stacking):

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_stacking, n_trials=50)

        # 最適なハイパーパラメータを表示
        print("Best parameters: ", study.best_params)

        return



