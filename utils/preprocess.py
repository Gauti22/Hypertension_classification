import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

scaler = joblib.load("/Users/gautammehta/Desktop/smal_project/transformer/scaler.pkl")

def preprocess_input(user_input_dict):
    df = pd.DataFrame([user_input_dict]) 
    # apply encoding and scaling here as you did during training
    wor=OrdinalEncoder()
    df['Exercise_Level']=wor.fit_transform(pd.DataFrame(df['Exercise_Level']))
    cat_col=df.select_dtypes('object').columns
    lc=LabelEncoder()
    for x in cat_col:
        df[x]=lc.fit_transform(df[x])
    return(df)

def scaled(user_input_dict):
    df=preprocess_input(user_input_dict)
    df_scaled = scaler.transform(df)
    
    return df_scaled