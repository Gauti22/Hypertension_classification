
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def data_read(path_x='/Users/gautammehta/Desktop/smal_project/data/processed/X_train_pro',path_y='/Users/gautammehta/Desktop/smal_project/data/processed/y_train_pro'):
    X=pd.read_csv(path_x)
    y=pd.read_csv(path_y)
    data=X.copy()
    data['y']=y
    return(data)


def encoding(data):
    wor=OrdinalEncoder()
    data['Exercise_Level']=wor.fit_transform(pd.DataFrame(data['Exercise_Level']))
    cat_col=data.select_dtypes('object').columns
    lc=LabelEncoder()
    for x in cat_col:
        data[x]=lc.fit_transform(data[x])
    return(data)


def scaling(data):
    X=data.iloc[:,:10]
    y=data['y']
    dt=StandardScaler()
    finl=dt.fit_transform(X)
    finl=pd.DataFrame(finl,columns=X.columns)
    finl['y']=data['y']
    return(finl)


def file_save(finl):
    path=os.path.join('/Users/gautammehta/Desktop/smal_project/data','EDA')
    os.makedirs(path,exist_ok=True)
    finl.iloc[:,:10].to_csv(os.path.join(path,'X_EDA'),index=False)
    finl['y'].to_csv(os.path.join(path,'y_EDA'),index=False)


if __name__=='__main__':
    df=data_read()
    df=encoding(df)
    df=scaling(df)
    file_save(df)