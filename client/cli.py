

import time
from typing import List

import rich
import typer
from rich.progress import track
from torch.utils.data import Dataset

from ..config import MainCfg
from ..data.loader import load_csv
from ..data.split import split_clients
from ..utils.logging import init_logger
from .core import Client, LocalDS  # 显式导入 LocalDS

app = typer.Typer(help="EAFL‑ALP 客户端 CLI")


@app.command()
def run(
    csv: str = "geolife.csv",
    num_clients: int = 5,
):

    cfg = MainCfg(csv_file=csv)
    cfg.training.batch_size =  256
    init_logger()

    # 1) 载入数据
    X, y, n_cls, _ = load_csv(cfg.csv_file)
    base_ds: Dataset = LocalDS(X, y)
    subsets: List[Dataset] = split_clients(base_ds, num_clients)

    # 2)
    Hmin = 0.0
    Hmax = cfg.quant.rho
    clients = [
        Client(i, subsets[i], n_cls, Hmin, Hmax, cfg)
        for i in range(num_clients)
    ]

    # 3)
    for rd in track(
        range(1, cfg.training.rounds + 1),
        description="Rounds",
        transient=True,
    ):
        raw_bytes = comp_bytes = 0
        accs: List[float] = []
        f1s:  List[float] = []

        for cli in clients:
            r, c, m = cli.step(rd)
            raw_bytes  += r
            comp_bytes += c
            accs.append(m["acc"])
            f1s.append(m["f1"])

        saved = raw_bytes - comp_bytes
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        mean_f1  = sum(f1s)  / len(f1s)  if f1s  else 0.0

        rich.print(
            f"[bold green]R{rd:02d}[/] "
            f"saved {saved:,} bytes (raw {raw_bytes:,} → comp {comp_bytes:,}) "
            f"acc={mean_acc:.3f}  f1={mean_f1:.3f}"
        )
        time.sleep(0.3)


if __name__ == "__main__":
    app()
