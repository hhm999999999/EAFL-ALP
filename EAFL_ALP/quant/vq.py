import numpy as np, math, os, struct
from sklearn.cluster import KMeans

BLOCK       = 256
RHO         = 0.2
CODES       = 256
ETA_STEP    = 1/256
VQ_BITS     = 8
LAPLACE_B   = 0.1
VQ_SEED     = 42
TUPLE_LEN   = 11

CODE_PATH   = "codebook.npy"
if os.path.exists(CODE_PATH):
    CODEBOOK = np.load(CODE_PATH).astype(np.float32)
else:
    samp = np.random.randn(50000, BLOCK).astype(np.float32)
    CODEBOOK = KMeans(CODES, n_init=8,
                      random_state=VQ_SEED).fit(samp).cluster_centers_.astype(np.float32)
    np.save(CODE_PATH, CODEBOOK)

def pack_flags(v):
    flags = 0
    for g in range(32):
        s = np.sign(v[g*8:(g+1)*8].sum())
        bit = 1 if s > 0 else 0
        flags |= (bit << g)
    return flags

def expand_flags(flags):
    s = np.zeros(BLOCK, np.float32)
    for g in range(32):
        bit = (flags >> g) & 1
        val = 1.0 if bit else -1.0
        s[g*8:(g+1)*8] = val
    return s

def encode_block(vec, bid):
    """返回 (tuple_bytes, quantised_vec)"""
    s = np.max(np.abs(vec)) / ((2**(VQ_BITS-1))-1) + 1e-8
    u = vec / s
    cid = int(np.argmin(((CODEBOOK - u)**2).sum(1)))
    flags = pack_flags(u - CODEBOOK[cid])


    rng = np.random.RandomState(VQ_SEED + bid)
    dith = rng.uniform(-0.5, 0.5, BLOCK).astype(np.float32)
    noise = np.random.laplace(0, LAPLACE_B, BLOCK).astype(np.float32)
    qt = (CODEBOOK[cid] + ETA_STEP * expand_flags(flags) + dith + noise) * s


    tup = struct.pack(
        "<I B I H",
        bid,
        cid,
        flags,
        np.float16(s).view("uint16")
    )
    return tup, qt

class VQQuant:
    def __init__(self, B=BLOCK, rho=RHO):
        self.B = B
        self.rho = rho


    def _pad(self, v):
        pad = (math.ceil(len(v)/self.B)*self.B) - len(v)
        return (np.concatenate([v, np.zeros(pad, np.float32)])
                if pad else v)

    def _layer(self, vec):
        vec = self._pad(vec)
        G = len(vec)//self.B
        blks = vec.reshape(G, self.B)
        var = blks.var(axis=1)
        K = int(math.ceil(self.rho*G))
        sel = np.argpartition(-var, K-1)[:K]

        payload = bytearray()
        for b in sel:
            tup, qt = encode_block(blks[b].copy(), b)
            blks[b] = qt
            payload.extend(tup)
        return payload

    def quant(self, delta, shared):
        pay = bytearray()
        idx = 0
        for lid, p in enumerate(shared.parameters()):
            n = p.numel()
            layer_pay = self._layer(delta[idx:idx+n])
            if layer_pay:
                pay.extend(struct.pack("<H", lid))
                pay.extend(struct.pack("<I", len(layer_pay)))
                pay.extend(layer_pay)
            idx += n
        return bytes(pay)
