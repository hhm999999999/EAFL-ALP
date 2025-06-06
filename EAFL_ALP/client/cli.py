import typer, time, rich
from rich.progress import track
from ..config import MainCfg
from ..data.loader import load_csv
from ..data.split  import split_clients
from .core import Client
from ..utils.logging import init_logger

app = typer.Typer(help="EAFL-ALP 客户端 CLI")

@app.command()
def run(csv: str = "geolife.csv", num_clients: int = 5):
    cfg = MainCfg(csv_file=csv)
    init_logger()

    # 1. 载入数据
    X, y, n_cls, df = load_csv(cfg.csv_file)
    from torch.utils.data import Dataset
    base_ds: Dataset = Client.LocalDS(X, y)  # 复用你在 core.py 里的类
    subs = split_clients(base_ds, num_clients)

    # 2. 构建客户端
    clients = [Client(i, subs[i], n_cls, cfg) for i in range(num_clients)]

    # 3. 联邦循环
    for rd in track(range(1, cfg.training.rounds + 1), description="Rounds"):
        raw, comp = 0, 0
        for cli in clients:
            bytes_raw, bytes_comp = cli.step(rd)
            raw += bytes_raw; comp += bytes_comp
        rich.print(f"[bold green]R{rd:02d}[/] saved {raw-comp:,.0f} bytes")
        time.sleep(0.3)

if __name__ == "__main__":
    app()
