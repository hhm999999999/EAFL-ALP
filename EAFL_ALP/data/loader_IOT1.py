import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
def load_iot1(path,date_fmt='%d-%b-%y',time_fmt='%H:%M:%S'):
    df=pd.read_csv(path)
    df.rename(columns={'date':'Date','time':'Time','latitude':'Latitude','longitude':'Longitude','label':'Label','type':'Type'},inplace=True)
    df['Date_Time']=pd.to_datetime(df['Date'].str.strip()+' '+df['Time'].str.strip(),format=f"{date_fmt} {time_fmt}",errors='coerce')
    df.dropna(subset=['Date_Time'],inplace=True)
    df=df[df['Latitude'].between(-90,90)&df['Longitude'].between(-180,180)]
    df['Hour']=df['Date_Time'].dt.hour
    df['Minute']=df['Date_Time'].dt.minute
    df['Hour_sin']=np.sin(2*np.pi*df['Hour']/24)
    df['Hour_cos']=np.cos(2*np.pi*df['Hour']/24)
    df['Min_sin']=np.sin(2*np.pi*df['Minute']/60)
    df['Min_cos']=np.cos(2*np.pi*df['Minute']/60)
    cols=['Latitude','Longitude','Hour_sin','Hour_cos','Min_sin','Min_cos']
    X=MinMaxScaler().fit_transform(df[cols].astype(np.float32).values)
    labs=df['Type'].unique().tolist()
    lmap={l:i for i,l in enumerate(labs)}
    df['TypeInt']=df['Type'].map(lmap)
    y=df['TypeInt'].astype(np.int64).values
    return X,y,len(labs),df
