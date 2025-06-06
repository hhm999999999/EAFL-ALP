from __future__ import annotations

import numpy as np

from .emam import emam, EMA   # global EMA 位于 emam.py 内


class Aggregator:



    def __init__(self, theta_init: np.ndarray):

        self.theta: np.ndarray = theta_init.copy()
        self.round: int        = 0


        global EMA
        EMA = self.theta.copy()

    # ---------------------------- 核心接口 --------------------------- #
    def apply_delta(self, delta: np.ndarray, delay: int) -> np.ndarray:


        self.theta = emam(self.theta, delta, delay)


        self.round += 1
        return self.theta.copy()

    def state_dict(self) -> dict:

        return {"theta": self.theta.copy(), "round": self.round}

    def load_state_dict(self, state: dict):

        self.theta = state["theta"].copy()
        self.round = int(state["round"])
        global EMA
        EMA = self.theta.copy()
