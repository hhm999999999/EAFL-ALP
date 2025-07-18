"""生成 / 读取 8-bit KMeans 码本。"""
import numpy as np, os
from sklearn.cluster import KMeans

def get_codebook(path: str = "codebook.npy",
                 codes: int = 256, block: int = 256,
                 seed: int = 42):
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    samp = np.random.randn(50000, block).astype(np.float32)
    cb   = KMeans(codes, n_init=8, random_state=seed)\
            .fit(samp).cluster_centers_.astype(np.float32)
    np.save(path, cb)
    return cb
