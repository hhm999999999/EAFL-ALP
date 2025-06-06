import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
def load_proxy(path):
    df=pd.read_csv(path)
    df.dropna(subset=['device_category'],inplace=True)
    feat_cols=[c for c in df.columns if c!='device_category']
    X=MinMaxScaler().fit_transform(df[feat_cols].astype(np.float32).values)
    df['LabelInt']=df['device_category'].astype('category').cat.codes
    y=df['LabelInt'].astype(np.int64).values
    return X,y
def load_client(path,epsilon=5,dp_seed=7777):
    df=pd.read_csv(path)
    df.dropna(subset=['device_category'],inplace=True)
    feat_cols=[c for c in df.columns if c!='device_category']
    feats=MinMaxScaler().fit_transform(df[feat_cols].astype(np.float32).values)
    rng=np.random.RandomState(dp_seed)
    feats+=rng.laplace(0,1.0/epsilon,feats.shape).astype(np.float32)
    feats=np.clip(feats,0.0,1.0)
    df['LabelInt']=df['device_category'].astype('category').cat.codes
    y=df['LabelInt'].astype(np.int64).values
    n_cls=len(df['device_category'].astype('category').cat.categories)
    return feats,y,n_cls,df
