
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import yaml
import json


mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Logistic_regression')

def train_data_X():
    X=pd.read_csv('/Users/gautammehta/Desktop/smal_project/data/EDA/X_EDAtrain')
    return(X)

def train_data_y():
    y=pd.read_csv('/Users/gautammehta/Desktop/smal_project/data/EDA/y_EDAtrain')
    return(y)

def test_data_X():
    X_test=pd.read_csv('/Users/gautammehta/Desktop/smal_project/data/EDA/X_EDAtest')
    return(X_test)

def test_data_y():
    y_test=pd.read_csv('/Users/gautammehta/Desktop/smal_project/data/EDA/y_EDAtest')
    return(y_test)

def model_build(X,y,X_test,y_test):
    with mlflow.start_run():
        signature = infer_signature(X_test, y_test)
        lr=LogisticRegression()
        lr.fit(X,y)
        y_pred=lr.predict(X_test)
        cm=confusion_matrix(y_test,y_pred)
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)

        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('precision',precision)
        mlflow.log_metric('recall',recall)
        mlflow.sklearn.log_model(lr, "logistic_model",signature=signature)


        mlflow.set_tag('author','gautam')
        mlflow.set_tag('model_type','Logistic_reg')

        metrics_dict = {
        "confusion_matrix": cm.tolist(),  # Convert NumPy array to list
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall }

        # Convert to JSON string
        metrics_json = json.dumps(metrics_dict, indent=4)

        path=os.path.join('/Users/gautammehta/Desktop/smal_project/reports','model')
        os.makedirs(path,exist_ok=True)
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            f.write(metrics_json)

if __name__=='__main__':
    X=train_data_X()
    y=train_data_y()
    X_test=test_data_X()
    y_test=test_data_y()
    model_build(X,y,X_test,y_test)






# 