import numpy as np,math,os
from sklearn.cluster import KMeans
BLOCK=256
RHO=0.20
CODES=256
VQ_BITS=8
ETA_STEP=1/256
LAPLACE_B=0.02
SEED=42
CODE_PATH="codebook.npy"
if os.path.exists(CODE_PATH):
    CODEBOOK=np.load(CODE_PATH).astype(np.float32)
else:
    samp=np.random.randn(50000,BLOCK).astype(np.float32)
    CODEBOOK=KMeans(CODES,n_init=8,random_state=SEED).fit(samp).cluster_centers_.astype(np.float32)
    np.save(CODE_PATH,CODEBOOK)
def pack(sign):
    flags=0
    for g in range(8):
        s=np.sign(sign[g*32:(g+1)*32].sum())
        code=0 if s==0 else(1 if s>0 else 2)
        flags|=(code<<(2*g))
    return int(flags)
def encode(vec,bid):
    s=np.max(np.abs(vec))/(2**(VQ_BITS-1)-1)+1e-8
    u=vec/s
    cid=int(np.argmin(((CODEBOOK-u)**2).sum(1)))
    sign=np.sign(u-CODEBOOK[cid]).astype(np.int8)
    dith=np.random.RandomState(SEED+bid).uniform(-0.5,0.5,BLOCK).astype(np.float32)
    noise=np.random.laplace(0,LAPLACE_B,BLOCK).astype(np.float32)
    qt=(CODEBOOK[cid]+ETA_STEP*sign+dith+noise)*s
    return cid,float(s),pack(sign),qt
class VQQuant:
    def __init__(self,B=BLOCK,rho=RHO):
        self.B=B
        self.rho=rho
    def _pad(self,v):
        pad=(math.ceil(len(v)/self.B)*self.B)-len(v)
        return np.concatenate([v,np.zeros(pad,np.float32)]) if pad else v
    def _layer(self,vec):
        vec=self._pad(vec)
        G=len(vec)//self.B
        blks=vec.reshape(G,self.B)
        var=blks.var(axis=1)
        K=int(math.ceil(self.rho*G))
        sel=np.argpartition(-var,K-1)[:K]
        meta=[]
        for b in sel:
            cid,s,flags,qt=encode(blks[b].copy(),b)
            blks[b]=qt
            meta.append({"block":int(b),"code":cid,"scale":s,"flags":flags})
        return meta
    def quant(self,delta,shared):
        metas=[]
        idx=0
        for lid,p in enumerate(shared.parameters()):
            n=p.numel()
            ml=self._layer(delta[idx:idx+n])
            for m in ml:
                m.update({"layer":lid})
            metas.extend(ml)
            idx+=n
        return metas
