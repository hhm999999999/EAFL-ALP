"""expand / decode_meta / emam —— 完整实现"""
import math, numpy as np
from functools import lru_cache
from .proxy_grad import proxy_grad
from ..quant.vq import CODEBOOK, BLOCK, ETA_STEP


LAMBDA_DELAY = 2e-3
ALPHA, ETA, BETA, MU = 0.8, 1e-3, 0.2, 0.95


EMA = None  # type: ignore[assignment]



@lru_cache(maxsize=2048)
def expand(flags: int):
    m = {0: 0.0, 1: 1.0, 2: -1.0, 3: 0.0}
    s = np.zeros(BLOCK, np.float32)
    for g in range(8):
        code = (flags >> (2 * g)) & 0b11
        s[g * 32 : (g + 1) * 32] = m[code]
    return s


def decode_meta(meta):


    from ..server.app import OFFS, TOTAL
    delta = np.zeros(TOTAL, np.float32)
    for m in meta:
        lid, bid = m['layer'], m['block']
        cid, sc  = m['code'],  m['scale']
        flags    = m['flags']

        blk = (CODEBOOK[cid] + ETA_STEP * expand(flags)) * sc
        st  = OFFS[lid] + bid * BLOCK
        ed  = min(OFFS[lid + 1], st + BLOCK)
        delta[st:ed] = blk[: ed - st]
    return delta



def emam(theta_old: np.ndarray, delta: np.ndarray, delay: int):
    """
    * θ_new = θ_old + α·w_del·δ - η·proxy_grad + β(EMA - θ_old)/(1+β)
    * EMA   = μ·EMA + (1-μ)·θ_new
    """
    global EMA
    if EMA is None:
        EMA = theta_old.copy()

    w_del   = math.exp(-LAMBDA_DELAY * delay)
    g_proxy = proxy_grad()

    theta = theta_old + (
        ALPHA * w_del * delta
        - ETA * g_proxy
        + BETA * (EMA - theta_old)
    ) / (1 + BETA)

    EMA = MU * EMA + (1 - MU) * theta
    return theta
