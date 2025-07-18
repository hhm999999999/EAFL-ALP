#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import base64
import json
import math
import struct
import requests
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

from ..models.combined import CombinedModel
from ..quant.vq import VQQuant
from ..config import MainCfg



class LocalDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def entropy_ds(ds: Dataset, bins: int = 30) -> float:
    arr = ds.dataset.X.numpy()[ds.indices] if isinstance(ds, Subset) else ds.X.numpy()
    h = []
    for j in range(arr.shape[1]):
        p, _ = np.histogram(arr[:, j], bins=bins, density=True)
        p += 1e-9
        h.append(-np.sum(p * np.log2(p)))
    return float(np.mean(h))


def width_factor(H: float, Hmin: float, Hmax: float,
                 wmin: float = 0.5, wmax: float = 1.0) -> float:
    return wmin + (H - Hmin) / (Hmax - Hmin + 1e-9) * (wmax - wmin)


def depth_factor(H: float) -> int:
    return 1 if H < 2.0 else (2 if H < 2.7 else (3 if H < 3.3 else 4))


def flat_shared(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel()
                           for p in model.shared.parameters()])


def load_shared(model: nn.Module, vec: np.ndarray) -> None:
    idx = 0
    with torch.no_grad():
        for p in model.shared.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[idx : idx + n]).view_as(p))
            idx += n



class Client:
    def __init__(
        self,
        cid: int,
        ds: Dataset,
        n_cls: int,
        Hmin: float,
        Hmax: float,
        cfg: MainCfg,
        split: float = 0.8,
    ):
        self.cid   = cid
        self.dev   = cfg.device
        self.cfg   = cfg
        self.vq    = VQQuant()


        H  = entropy_ds(ds)
        w  = width_factor(H, Hmin, Hmax)
        D  = depth_factor(H)


        sample_loader = DataLoader(ds, batch_size=cfg.training.batch_size,
                                   shuffle=True)


        self.m = CombinedModel(n_cls, w, D, cid,
                               sample_loader=sample_loader).to(self.dev)
        self.loss = nn.CrossEntropyLoss()
        self.opt  = optim.AdamW(
            self.m.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )


        n_tr = int(len(ds) * split)
        tr, val = random_split(
            ds,
            [n_tr, len(ds) - n_tr],
            generator=torch.Generator().manual_seed(cid),
        )
        self.trL  = DataLoader(tr,  cfg.training.batch_size,     shuffle=True)
        self.valL = DataLoader(val, cfg.training.batch_size * 2, shuffle=False)


    def train(self, theta: np.ndarray, epochs: int):
        load_shared(self.m, theta)
        self.m.train()
        for _ in range(epochs):
            for xb, yb in self.trL:
                xb, yb = xb.to(self.dev), yb.to(self.dev)
                self.opt.zero_grad()
                self.loss(self.m(xb), yb).backward()
                self.opt.step()


    def step(self, round_idx: int) -> Tuple[int, int, dict]:
        theta = self.pull_theta()
        self.train(theta, self.cfg.training.local_epochs)
        delta = flat_shared(self.m) - theta


        nrm = np.linalg.norm(delta, 1)
        if nrm > 1.0:
            delta = delta / nrm

        rawB = delta.nbytes
        pay_bin = self.vq.quant(delta.copy(), self.m.shared)

        # —— 本地验证集评估 ———————————————— ★ add
        self.m.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in self.valL:
                logits = self.m(xb.to(self.dev)).argmax(1).cpu().numpy()
                y_pred.extend(logits)
                y_true.extend(yb.numpy())
        metrics = {
            "acc": float(accuracy_score(y_true, y_pred)),
            "f1" : float(f1_score(y_true, y_pred, average="macro"))
        }
        # ————————————————————————————————————

        body = {
            "cid"      : self.cid,
            "t"        : round_idx,
            "srv_round": round_idx,
            "bin"      : base64.b64encode(pay_bin).decode(),
            "metrics"  : metrics,
        }
        requests.post(
            f"{self.cfg.server_url}/upload",
            json=body,
            timeout=120,
        )
        return rawB, len(pay_bin), metrics


    def pull_theta(self) -> np.ndarray:
        r = requests.get(f"{self.cfg.server_url}/params", timeout=60).json()
        return np.frombuffer(
            base64.b64decode(r["data"]), dtype=np.float32
        ).reshape(r["shape"])
