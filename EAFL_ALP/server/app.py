#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import math
import struct
import base64
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify


from ..models.fractal import GlobalShared
from ..federation.emam import emam

# ───────────── 量化常量 ──────────────
BLOCK        = 256        # 向量量化块长
CODES        = 256        # codebook 大小
ETA_STEP     = 1 / 256    # dithering 步长
TUPLE_LEN    = 11         # 88 bit 元组长度 (byte)
CODE_PATH    = "codebook.npy"


if os.path.exists(CODE_PATH):
    CODEBOOK = np.load(CODE_PATH).astype(np.float32)
else:
    from sklearn.cluster import KMeans
    samp = np.random.randn(50_000, BLOCK).astype(np.float32)
    CODEBOOK = (
        KMeans(CODES, n_init=8, random_state=42)
        .fit(samp)
        .cluster_centers_.astype(np.float32)
    )
    np.save(CODE_PATH, CODEBOOK)


DEVICE       = "cpu"
net          = GlobalShared().to(DEVICE)
def flat_param(m: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in m.parameters()])

def load_vec(m: nn.Module, vec: np.ndarray) -> None:
    idx = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[idx : idx + n]).view_as(p))
            idx += n

THETA  = flat_param(net)
NUMS   = [p.numel() for p in net.parameters()]
OFFS   = np.cumsum([0] + NUMS)
TOTAL  = OFFS[-1]


def expand(flags: int) -> np.ndarray:
    s = np.zeros(BLOCK, np.float32)
    for g in range(32):
        bit = (flags >> g) & 1
        val = 1.0 if bit else -1.0
        s[g * 8 : (g + 1) * 8] = val
    return s


def apply_layer(delta: np.ndarray, lid: int, blk_bytes: bytes) -> None:

    for off in range(0, len(blk_bytes), TUPLE_LEN):
        bid, cid, flags, s16 = struct.unpack_from("<I B I H", blk_bytes, off)
        scale = np.float16(s16).item()
        blk = (CODEBOOK[cid] + ETA_STEP * expand(flags)) * scale
        st  = OFFS[lid] + bid * BLOCK
        ed  = min(OFFS[lid + 1], st + BLOCK)
        delta[st:ed] = blk[: ed - st]


app = Flask(__name__)
PORT = 8080

@app.route("/params", methods=["GET"])
def get_params():

    return jsonify(
        {
            "shape": [int(x) for x in THETA.shape],
            "data" : base64.b64encode(THETA.astype(np.float32).tobytes()).decode(),
            "ts"   : int(datetime.utcnow().timestamp()),
        }
    )

@app.route("/upload", methods=["POST"])
def upload():

    global THETA
    js      = request.get_json(force=True)
    bin_raw = base64.b64decode(js["bin"])
    delta   = np.zeros(TOTAL, np.float32)

    ptr = 0
    while ptr < len(bin_raw):
        lid   = struct.unpack_from("<H", bin_raw, ptr)[0] ; ptr += 2
        ln    = struct.unpack_from("<I", bin_raw, ptr)[0] ; ptr += 4
        apply_layer(delta, lid, bin_raw[ptr : ptr + ln])
        ptr  += ln

    # 异步能量聚合
    THETA = emam(THETA, delta, delay=0)
    load_vec(net, THETA)
    return jsonify({"ok": True})

if __name__ == "__main__":
    print(f"[{datetime.now()}] EAFL-ALP server PARAM={TOTAL/1e3:.1f}k")

    app.run("0.0.0.0", PORT, threaded=True)
