import numpy as np,torch,torch.nn as nn
from ..data.loader import load_unsw
from ..models.fractal import GlobalShared
CSV="VOI.csv"
X_raw,y_raw,_,_=load_unsw(CSV,'attack_cat')
idx=np.random.choice(len(X_raw),int(0.2*len(X_raw)),replace=False)
PX=torch.tensor(X_raw[idx],dtype=torch.float32)
PY=torch.tensor(y_raw[idx],dtype=torch.long)
net=GlobalShared(X_raw.shape[1])
loss_fn=nn.CrossEntropyLoss()
def proxy_grad():
    net.train()
    net.zero_grad()
    loss=loss_fn(net(PX),PY)
    loss.backward()
    return np.concatenate([p.grad.detach().numpy().ravel() for p in net.parameters()])
