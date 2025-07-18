import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
def load_unsw(path,target='attack_cat'):
    df=pd.read_csv(path)
    if target=='attack_cat':
        df[target].replace({'-':'Normal'},inplace=True)
        df[target].fillna('Normal',inplace=True)
        y_raw=df[target].astype(str)
        classes=sorted(y_raw.unique())
        lmap={c:i for i,c in enumerate(classes)}
        labels=y_raw.map(lmap).astype(np.int64).values
        n_cls=len(classes)
    elif target=='label':
        labels=df['label'].astype(np.int64).values
        n_cls=2
    else:
        raise ValueError
    drop_cols={target}
    if 'label' in df.columns:
        drop_cols.add('label')
    feat_cols=[c for c in df.columns if c not in drop_cols]
    cat_cols=[c for c in feat_cols if df[c].dtype=='object']
    for c in cat_cols:
        df[c]=pd.factorize(df[c])[0].astype(np.int32)
    feats=df[feat_cols].astype(np.float32).values
    feats=MinMaxScaler().fit_transform(feats).astype(np.float32)
    return feats,labels,n_cls,df
