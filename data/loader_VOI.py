import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
def load_proxy(path):
    df=pd.read_csv(path)
    df['Global_Time']=pd.to_datetime(df['Global_Time'],unit='ms',errors='coerce')
    df.sort_values(['Vehicle_ID','Global_Time'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    df['dt']=df.groupby('Vehicle_ID')['Global_Time'].diff().dt.total_seconds().fillna(0)
    df['dX']=df.groupby('Vehicle_ID')['Local_X'].diff().fillna(0)
    df['dY']=df.groupby('Vehicle_ID')['Local_Y'].diff().fillna(0)
    cols=['Local_X','Local_Y','Global_X','Global_Y','v_length','v_Width','v_Vel','v_Acc','Space_Headway','Time_Headway','dt','dX','dY']
    X=StandardScaler().fit_transform(df[cols].fillna(0).astype(np.float32).values)
    df['LabelInt']=df['v_Class'].astype('category').cat.codes
    y=df['LabelInt'].astype(np.int64).values
    return X,y
def load_client(path):
    df=pd.read_csv(path)
    df['Global_Time']=pd.to_datetime(df['Global_Time'],unit='ms',errors='coerce')
    df.sort_values(['Vehicle_ID','Global_Time'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    df['dt']=df.groupby('Vehicle_ID')['Global_Time'].diff().dt.total_seconds().fillna(0)
    df['dX']=df.groupby('Vehicle_ID')['Local_X'].diff().fillna(0)
    df['dY']=df.groupby('Vehicle_ID')['Local_Y'].diff().fillna(0)
    cols=['Local_X','Local_Y','Global_X','Global_Y','v_length','v_Width','v_Vel','v_Acc','Space_Headway','Time_Headway','dt','dX','dY']
    X=StandardScaler().fit_transform(df[cols].fillna(0).astype(np.float32).values)
    df['LabelInt']=df['v_Class'].astype('category').cat.codes
    y=df['LabelInt'].astype(np.int64).values
    n_cls=len(df['v_Class'].astype('category').cat.categories)
    return X,y,n_cls,df
