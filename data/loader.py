import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_csv(path: str, fmt: str = "%Y-%m-%d %H:%M:%S",
             dp_b: float | None = None, rng_seed: int = 7777):
    df = pd.read_csv(path)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], format=fmt, errors="coerce")
    df = df.dropna(subset=["Date_Time"])
    df = df[(df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))]


    if dp_b is not None:
        rng = np.random.RandomState(rng_seed)
        df["Latitude"]  += rng.laplace(0, dp_b, size=len(df))
        df["Longitude"] += rng.laplace(0, dp_b, size=len(df))
        df["Latitude"]  = df["Latitude"].clip(-90,  90)
        df["Longitude"] = df["Longitude"].clip(-180, 180)

    df["Hour"]   = df["Date_Time"].dt.hour
    df["Minute"] = df["Date_Time"].dt.minute
    df["Hour_sin"] = np.sin(2*np.pi*df["Hour"]/24)
    df["Hour_cos"] = np.cos(2*np.pi*df["Hour"]/24)
    df["Min_sin"]  = np.sin(2*np.pi*df["Minute"]/60)
    df["Min_cos"]  = np.cos(2*np.pi*df["Minute"]/60)

    cols = ["Latitude", "Longitude", "Altitude",
            "Hour_sin", "Hour_cos", "Min_sin", "Min_cos"]
    feats = MinMaxScaler().fit_transform(df[cols].astype(np.float32).values)

    labels  = df["Label"].unique().tolist()
    mapping = {l: i for i, l in enumerate(labels)}
    df["LabelInt"] = df["Label"].map(mapping)
    y = df["LabelInt"].astype(np.int64).values
    return feats, y, len(labels), df
