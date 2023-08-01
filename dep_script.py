import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import mlflow
import sklearn
import scipy
import xgboost as xgb
import joblib
import pathlib


from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("postgresql://mandeebot:pass@localhost:5432/mandeebot")
mlflow.set_experiment("Fake-News-Exp")
mlflow.autolog()


@task(retries=3)
def read_data(file):
    data = pd.read_csv(file)
    return data

@task
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

@task
def add_jaccard_similarity(data):
    count = 0
    for i in tqdm(range(data.shape[0])):
        jaccard_lis = []
        eps = 0.001
        sentence = data.loc[i, 'content'].split('.')  # per sentence scorer
        for j in range(len(sentence)):
            jaccard_lis.append(jaccard_similarity(data.loc[i, 'title'].split(' '), sentence[j].split(' ')))
        max_jaccard_similarity = max(jaccard_lis)
        avg_jaccard_similarity = sum(jaccard_lis) / len(jaccard_lis)
        min_jaccard_similarity = min(jaccard_lis)
        data.loc[i, 'jaccard_similarity'] = (max_jaccard_similarity + min_jaccard_similarity) / (max_jaccard_similarity - min_jaccard_similarity + eps)
    return data

@task
def feat_neng(data):

    x = data.iloc[:, -1]
    y = data['stance_num']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle=True)

    return x_train, x_test, y_train, y_test


@task(log_prints=True)       
def train_xg(x_train: scipy.sparse._csr.csr_matrix,
    x_test: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> None:
    
    x_train = x_train.values.reshape(-1,1)
    x_test = x_test.values.reshape(-1,1)
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "seed": 42,
        }
        xg_clf = xgb.XGBClassifier(best_params)


        mlflow.log_params(best_params)

        xg_clf.fit(x_train,y_train)

        y_pred = xg_clf.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        
        print('Accuracy: %f' % accuracy)

        # precision tp / (tp + fp)
        precision = precision_score(y_test, y_pred,average='micro')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_pred, y_test,average='micro')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_pred, y_test,average='micro')
        print('F1 score: %f' % f1)
        
        mlflow.log_metrics({"F1 score": f1,
            "Precision":precision,
             "Recall": recall })

        with open("models/preprocessor.b", "wb") as f_out:
            joblib.dump(xg_clf, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor2")

        #mlflow.xgboost.log_model(xg_clf, artifact_path="models_mlflow")
    return None
    # data.to_csv("updated_data.csv", index=False)

@flow
def main(
        train_path: str = '/Users/mandeebot/Desktop/fin_project/data_combined.csv'
) -> None:
    data = read_data(train_path)

    # jaccard_similarity(list1, list2)
    add_jaccard_similarity(data)

    x_train, x_test, y_train, y_test = feat_neng(data)
    train_rf(x_train, x_test, y_train, y_test)
    train_xg(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()