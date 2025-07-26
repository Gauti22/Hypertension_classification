import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def read_data(path):
    df=pd.read_csv(path)
    return(df)

def fill_missing(df):
    df.fillna('No',inplace=True)
    return(df)

def transfer_files(df):
    X=df.iloc[:,:10]
    y=df['Has_Hypertension']
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)

    path=os.path.join('/Users/gautammehta/Desktop/smal_project/data','processed')
    os.makedirs(path,exist_ok=True)

    X_train.to_csv(os.path.join(path,'X_train_pro'),index=False)
    X_test.to_csv(os.path.join(path,'X_test_pro'),index=False)
    y_test.to_csv(os.path.join(path,'y_test_pro'),index=False)
    y_train.to_csv(os.path.join(path,'y_train_pro'),index=False)


if __name__=='__main__':
    df=read_data('/Users/gautammehta/Downloads/hypertension_dataset.csv')
    df=fill_missing(df)
    transfer_files(df)
    
    
