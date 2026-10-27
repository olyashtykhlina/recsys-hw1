"""Item-based kNN: косинусное сходство со сжатием (shrinkage) и отбором k ближайших соседей."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .utils import BaseRecommender, Stage


class ItemKNN(BaseRecommender):
    """s_ij = <x_i, x_j> / (||x_i|| ||x_j|| + shrink),  score(u, i) = sum_{j in I(u) ∩ N_k(i)} s_ij.

    x_i — бинарный столбец матрицы взаимодействий (кто взаимодействовал с объектом i),
    N_k(i) — k объектов с наибольшим сходством с i.
    """

    name = "ItemKNN"

    def __init__(self, k: int = 100, shrink: float = 10.0, block: int = 2048):
        self.k = k
        self.shrink = shrink
        self.block = block

    def fit(self, stage: Stage):
        X = stage.X.astype(np.float32)
        self.X = X
        co = (X.T @ X).toarray()                               # n_items x n_items, co[i, j] = |U_i ∩ U_j|
        norms = np.sqrt(np.diag(co)).astype(np.float32)
        n = co.shape[0]
        k = min(self.k, n - 1)
        rows, cols, vals = [], [], []
        for s in range(0, n, self.block):                      # по блокам, чтобы не держать вторую n x n матрицу
            sim = co[s:s + self.block] / (norms[s:s + self.block, None] * norms[None, :] + self.shrink + 1e-12)
            sim[np.arange(sim.shape[0]), np.arange(s, s + sim.shape[0])] = 0.0   # без самоподобия
            nn = np.argpartition(-sim, k, axis=1)[:, :k]                         # соседи объекта i
            v = np.take_along_axis(sim, nn, axis=1)
            rows.append(np.repeat(np.arange(s, s + sim.shape[0]), k))
            cols.append(nn.ravel())
            vals.append(v.ravel())
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))
        keep = vals > 0
        # W[j, i] = s_ij, если j входит в соседи i: тогда score_u = x_u @ W
        self.W = sp.csr_matrix((vals[keep], (cols[keep], rows[keep])), shape=(n, n))
        return self

    def score(self, users):
        return (self.X[users] @ self.W).toarray()
