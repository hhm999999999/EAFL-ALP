# EAFL-ALP: Energy-Efficient Asynchronous Federated Learning with Adaptive Layered Personalization for Vehicular Networks

EAFL‑ALP tackles three long‑standing challenges in large‑scale Industrial‑IoT (IIoT) FL: non‑IID data, communication bottlenecks, and asynchronous instability. It integrates an Adaptive Fractal‑Wave Personalization Model (AFWPM), an 88‑bit Layer‑wise Quantization‑based Reversible DP Codec (LQGCM), and an Energy‑Minimization Aggregation Model (EMAM). On five IIoT benchmarks, EAFL‑ALP achieves up to 32.1 % higher accuracy, 3.3× faster convergence, and 99.8 % lower traffic than strong baselines.

This repository accompanies the paper **“EAFL‑ALP: Energy‑Efficient Asynchronous Federated Learning with Adaptive Layered Personalization for Vehicular Networks.”** 


---

## 1. Prerequisites

| Software | Version tested | Install link |
|----------|---------------:|--------------|
| Python   | 3.9 – 3.11 | <https://www.python.org> |
| Conda    | ≥ 4.10 *or* pip | <https://docs.conda.io> |
| Git      | any            | <https://github.com/hhm999999999/EAFL-ALP> |

---

## 2. Quick start

To ensure reproducibility, the exact versions of the runtime environment are recorded in requirements.txt 
```bash
 ① clone (or unzip) the repo
git clone https://github.com/hhm999999999/EAFL-ALP.git
cd EAFL-ALP-main               ← project root

 ② create & activate environment
conda env create -f env.yaml      or:  pip install -r requirements.txt
conda activate eafl-alp
```

---

## 3. Datasets

| Name (paper) | Source & licence |
|--------------|------------------|
| GeoLife v1.3 | <https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide> 
| NGSIM US‑101/I‑80 | <https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm>  
| TON_IoT | <https://research.unsw.edu.au/projects/toniot-datasets>  
| UNSW‑NB15 | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> 
| IoT‑Devices | <https://www.kaggle.com/code/davegn/iot-device-identification-classification/notebook> 

---

## 4. Single‑machine simulation guide (Sync vs Async)
This section shows two typical ways to replay EAFL‑ALP on one computer:


**Sync‑Batch**: launches N fixed clients, waits until all of them complete a round, then performs one global aggregation.

**Async‑V2X**:  mimics a vehicular network where clients appear or disappear at any moment and the server aggregates every gradient as soon as it arrives.



### Synchronous batch benchmark (fixed clients)

```bash
 ① start the asynchronous FL server (in background)
python EAFL_ALP/server/app.py --host 127.0.0.1 --port 8080 

 ② simulate N clients locally (choose a CSV listed above)
python -m EAFL_ALP.client.cli run --csv geolife.csv --num-clients 5
```

### Asynchronous V2X (dynamic clients)

```bash
 ① start the asynchronous FL server (in background)
python EAFL_ALP/server/app.py --host 127.0.0.1 --port 8080 

 ② simulate N clients locally (A few seconds later, start another client)
 sleep 5
python -m EAFL_ALP.client.cli run --csv geolife1-5.csv --num-clients 1
```

**Expected console output (excerpt)**  
```
Rounds ━━━━━━━━━━━━━━━━━ 100% 50/50 • 0:15 • 300ms/it
R01 saved 1 234 567 bytes (raw 1 345 132 → comp 110 565)
            ⋮
R50 saved 1 402 899 bytes (raw 1 445 376 → comp 142 477)
Final accuracy 0.902 | uplink compression 97.8%
```

---

## 5. Custom hyper‑parameters

All tunables live in `EAFL_ALP/config.py` and can be overridden on‑the‑fly (`--cfg.<field>=value`):

```python
class TrainingCfg:
    rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 5e-5

class QuantCfg:
    block: int = 256
    rho: float = 0.20
    codes: int = 256
    vq_bits: int = 8
    eta_step: float = 1 / 256

class MainCfg:
    server_url: str = "http://127.0.0.1:8080"
    csv_file: Path = Path("geolife.csv")
    seed: int = 42
    device: str = "cuda"
    training: TrainingCfg = field(default_factory=TrainingCfg)
    quant: QuantCfg = field(default_factory=QuantCfg)
```

---

## 6. Common issues & fixes

| Symptom | Resolution |
|---------|------------|
| `ImportError … relative import` | `export PYTHONPATH=$PWD`  (Windows → `set PYTHONPATH=%cd%`) |
| `Client.__init__() missing Hmax` | Launch via **`client/cli.py`** (not a raw script) |
| GPU OOM | Lower `--cfg.training.batch_size` or switch to CPU |
| Port 8080 busy | Start server with `--port 8090` and update `MainCfg.server_url` |

---

## 7. Protocol overview

```mermaid
sequenceDiagram
    participant S as Server (Flask)
    participant C1 as Client 1
    participant Cn as Client n

    C1->>S: GET /params (θ)
    Cn->>S: GET /params (θ)
    par Round r
        C1->>C1: Local SGD → Δ₁
        Cn->>Cn: Local SGD → Δₙ
        C1->>S: POST /upload (quantised Δ₁)
        Cn->>S: POST /upload (quantised Δₙ)
        S->>S: EMAM(θ, {Δₖ})
    end
    Note over S: θ updates online; clients never block
```

---

## 8. Project structure

```text
EAFL-ALP-main/
└── EAFL_ALP/            # Python package
    ├── client/          # device logic (training, quant, comms)
    ├── server/          # Flask aggregation service
    ├── federation/      # asynchronous EMAM 
    ├── models/          # shared backbone + local private heads
    ├── quant/           # low-bit VQ compression
    ├── data/            # CSV I/O & splits
    ├── utils/  | tests/ # helpers / unit tests
    └── scripts/         # run_server.sh / run_clients.sh
```


## 9. Acknowledgements
We would like to express our sincere gratitude to the anonymous reviewer for their insightful and constructive comments on our manuscript. Your keen observations have significantly enriched and strengthened our paper. We are truly grateful for the time and effort you dedicated to our work.
