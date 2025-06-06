import numpy as np
from torch.utils.data import Dataset, Subset

def split_clients(base: Dataset, k: int):
    n, step = len(base), len(base) // k
    subs = []
    for i in range(k):
        lo, hi = i * step, (i + 1) * step if i < k - 1 else n
        subs.append(Subset(base, np.arange(lo, hi)))
    return subs
