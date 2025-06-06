#!/usr/bin/env python3
import base64,math,numpy as np,torch,torch.nn as nn
from flask import Flask,request,jsonify
from functools import lru_cache
from datetime import datetime
from ..data.loader_iot import load_proxy
from ..models.fractal import GlobalShared
from ..quant.vq import CODEBOOK,BLOCK,ETA_STEP
from ..config import MainCfg
cfg=MainCfg()
X_raw,y_raw=load_proxy(cfg.csv_proxy)
idx=np.random.choice(len(X_raw),int(0.2*len(X_raw)),replace=False)
PX=torch.tensor(X_raw[idx],dtype=torch.float32)
PY=torch.tensor(y_raw[idx],dtype=torch.long)
net=GlobalShared(X_raw.shape[1]).to(cfg.device)
crit=nn.CrossEntropyLoss()
def flat(m):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in m.parameters()])
THETA=flat(net)
EMA=THETA.copy()
nums=[p.numel() for p in net.parameters()]
OFFS=np.cumsum([0]+nums)
ALPHA=1.0
ETA=1e-3
BETA=0.2
MU=0.9
LAMBDA=2e-3
def proxy_grad():
    net.train()
    net.zero_grad()
    loss=crit(net(PX.to(cfg.device)),PY.to(cfg.device))
    loss.backward()
    return np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in net.parameters()])
@lru_cache(maxsize=1024)
def expand(flags):
    m={0:0.,1:1.,2:-1.,3:0.}
    s=np.zeros(BLOCK,np.float32)
    for g in range(8):
        code=(flags>>(2*g))&0b11
        s[g*32:(g+1)*32]=m[code]
    return s
def decode(meta):
    delta=np.zeros(OFFS[-1],dtype=np.float32)
    for m in meta:
        lid=m['layer']
        bid=m['block']
        cid=m['code']
        sc=m['scale']
        fl=m['flags']
        vec=(CODEBOOK[cid]+ETA_STEP*expand(fl))*sc
        st=int(OFFS[lid]+bid*BLOCK)
        ed=min(OFFS[lid+1],st+BLOCK)
        delta[st:ed]=vec[:ed-st]
    return delta
def emam(theta,delta,delay):
    global EMA
    w=math.exp(-LAMBDA*delay)
    g=proxy_grad()
    theta_new=theta+(ALPHA*w*delta-ETA*g+BETA*(EMA-theta))/(1+BETA)
    EMA=MU*EMA+(1-MU)*theta_new
    return theta_new
b64=lambda a:base64.b64encode(a.astype(np.float32).tobytes()).decode()
app=Flask(__name__)
@app.route("/params")
def params():
    return jsonify({"shape":list(THETA.shape),"data":b64(THETA)})
@app.route("/upload",methods=['POST'])
def upload():
    global THETA
    js=request.get_json(force=True)
    delta=decode(js['meta'])
    THETA=emam(THETA,delta,0)
    return jsonify({"ok":True})
if __name__=="__main__":
    print(f"{datetime.now()} SERVER_IOT PARAM={OFFS[-1]/1e3:.1f}k")
    app.run("0.0.0.0",8080,threaded=True)
