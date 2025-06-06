import os, math, numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pathlib import Path

from ..models.fractal import GlobalShared


PORT            = 8080
CSV_FILE        = "11.csv"
BLOCK           = 256
CODES           = 256
ETA_STEP        = 1 / 256
SEED_DITHER     = 42
DEVICE          = "cpu"


def load_proxy(path, fmt='%Y-%m-%d %H:%M:%S'):
    df = pd.read_csv(path)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format=fmt, errors='coerce')
    df = df.dropna(subset=['Date_Time'])
    df = df[(df['Latitude'].between(-90, 90)) & (df['Longitude'].between(-180, 180))]


    df['Hour']   = df['Date_Time'].dt.hour
    df['Minute'] = df['Date_Time'].dt.minute
    df['Hour_sin'] = np.sin(2*np.pi*df['Hour']/24)
    df['Hour_cos'] = np.cos(2*np.pi*df['Hour']/24)
    df['Min_sin']  = np.sin(2*np.pi*df['Minute']/60)
    df['Min_cos']  = np.cos(2*np.pi*df['Minute']/60)

    cols = ['Latitude','Longitude','Altitude',
            'Hour_sin','Hour_cos','Min_sin','Min_cos']
    X = MinMaxScaler().fit_transform(df[cols].astype(np.float32).values)

    labels  = df['Label'].unique().tolist()
    mapping = {l:i for i,l in enumerate(labels)}
    df['LabelInt'] = df['Label'].map(mapping)
    y = df['LabelInt'].astype(np.int64).values
    return X, y

PX_raw, PY_raw = load_proxy(CSV_FILE)
proxy_idx = np.random.choice(len(PX_raw), int(0.2 * len(PX_raw)), replace=False)
PX = torch.tensor(PX_raw[proxy_idx], dtype=torch.float32)
PY = torch.tensor(PY_raw[proxy_idx], dtype=torch.long)


net  = GlobalShared().to(DEVICE)
crit = nn.CrossEntropyLoss()

def flat(m):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in m.parameters()])

def proxy_grad():
    net.train(); net.zero_grad()
    loss = crit(net(PX.to(DEVICE)), PY.to(DEVICE))
    loss.backward()
    return np.concatenate(
        [p.grad.detach().cpu().numpy().ravel() for p in net.parameters()]
    )
