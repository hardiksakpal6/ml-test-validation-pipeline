import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data()-> pd.DataFrame:
    ds=load_breast_cancer(as_frame=True)
    df=ds.frame
    df=df.rename(columns={"target":"label"})
    return df

def split_data(df:pd.DataFrame, test_size:float=0.2, seed: int=42):
    X=df.drop(columns=["label"])
    y=df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
