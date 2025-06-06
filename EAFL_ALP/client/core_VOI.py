import numpy as np,torch,torch.nn as nn,torch.optim as optim,base64,json,requests,math,random
from torch.utils.data import Dataset,DataLoader,Subset,random_split
from ..config import MainCfg
from ..data.loader_vissim import load_client
from ..data.split import split_time
from ..models.combined import CombinedModel
from ..quant.vq import VQQuant
class LocalDS(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,i):
        return self.X[i],self.y[i]
def entropy(ds):
    arr=ds.dataset.X.numpy()[ds.indices] if isinstance(ds,Subset) else ds.X.numpy()
    return float(arr.var(axis=0).mean())
def width_factor(H,Hmin,Hmax):
    return 0.5+(H-Hmin)/(Hmax-Hmin+1e-9)*(0.5)
def depth_factor(H):
    return 1 if H<0.05 else 2
def flat_shared(m):
    return np.concatenate([p.detach().numpy().ravel() for p in m.shared.parameters()])
def load_shared(m,vec):
    idx=0
    with torch.no_grad():
        for p in m.shared.parameters():
            n=p.numel()
            p.copy_(torch.from_numpy(vec[idx:idx+n]).view_as(p))
            idx+=n
class Client:
    def __init__(self,cid,ds,n_cls,Hmin,Hmax,cfg):
        self.cid=cid
        self.cfg=cfg
        H=entropy(ds)
        w=width_factor(H,Hmin,Hmax)
        D=depth_factor(H)
        self.model=CombinedModel(n_cls,w,D,cid).to(cfg.device)
        self.loss=nn.CrossEntropyLoss()
        self.opt=optim.AdamW(self.model.parameters(),lr=cfg.training.lr,weight_decay=cfg.training.weight_decay)
        n_tr=int(len(ds)*0.8)
        tr,val=random_split(ds,[n_tr,len(ds)-n_tr],generator=torch.Generator().manual_seed(cid))
        self.trL=DataLoader(tr,cfg.training.batch_size,shuffle=True)
        self.valL=DataLoader(val,cfg.training.batch_size*2,shuffle=False)
        self.vq=VQQuant()
    def train(self,theta):
        load_shared(self.model,theta)
        self.model.train()
        for xb,yb in self.trL:
            xb,yb=xb.to(self.cfg.device),yb.to(self.cfg.device)
            self.opt.zero_grad()
            self.loss(self.model(xb),yb).backward()
            self.opt.step()
    def delta(self,theta):
        return flat_shared(self.model)-theta
    def push(self,delta):
        metas=self.vq.quant(delta.copy(),self.model.shared)
        requests.post(f"{self.cfg.server_url}/upload",data=json.dumps({"cid":self.cid,"meta":metas}),headers={'Content-Type':'application/json'},timeout=120)
def pull_theta(cfg):
    r=requests.get(f"{cfg.server_url}/params").json()
    return np.frombuffer(base64.b64decode(r['data']),dtype=np.float32).reshape(r['shape'])
def run_vissim():
    cfg=MainCfg()
    X,y,n_cls,df=load_client(cfg.csv_client)
    base_ds=LocalDS(X,y)
    subs=split_time(base_ds,5)
    Hs=[entropy(s) for s in subs]
    Hmin,Hmax=min(Hs),max(Hs)
    clients=[Client(i,subs[i],n_cls,Hmin,Hmax,cfg) for i in range(5)]
    for _ in range(cfg.training.rounds):
        theta=pull_theta(cfg)
        for cli in clients:
            cli.train(theta)
            cli.push(cli.delta(theta))
