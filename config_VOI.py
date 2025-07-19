from dataclasses import dataclass,field
from pathlib import Path
@dataclass
class TrainingCfg:
    rounds:int=50
    local_epochs:int=1
    batch_size:int=256
    lr:float=3e-4
    weight_decay:float=5e-5
@dataclass
class QuantCfg:
    block:int=256
    rho:float=0.20
    codes:int=256
    vq_bits:int=8
    eta_step:float=1/256
@dataclass
class MainCfg:
    server_url:str="http://127.0.0.1:8080"
    csv_proxy:Path=Path("VOI.csv")
    csv_client:Path=Path("123aa_20percent.csv")
    seed:int=42
    device:str="cuda"
    training:TrainingCfg=field(default_factory=TrainingCfg)
    quant:QuantCfg=field(default_factory=QuantCfg)
