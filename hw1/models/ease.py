"""EASE (Steck, 2019): линейная item-item модель с замкнутым решением."""
from __future__ import annotations

import numpy as np
import scipy.linalg

from .utils import BaseRecommender, Stage


class EASE(BaseRecommender):
    """min_B ||X - XB||_F^2 + lam ||B||_F^2,  s.t. diag(B) = 0.

    Решение: P = (X^T X + lam I)^{-1},  B_ij = -P_ij / P_jj (i != j),  B_jj = 0.
    Скор: score_u = x_u B.
    """

    name = "EASE"

    def __init__(self, lam: float = 500.0):
        self.lam = lam

    def fit(self, stage: Stage):
        X = stage.X
        self.X = X
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        # G симметрична и положительно определена -> обращение через разложение Холецкого
        c, low = scipy.linalg.cho_factor(G, overwrite_a=True, check_finite=False)
        P = scipy.linalg.cho_solve((c, low), np.eye(G.shape[0]), overwrite_b=True, check_finite=False)
        del G, c
        B = P / (-np.diag(P))[None, :]
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.B)
